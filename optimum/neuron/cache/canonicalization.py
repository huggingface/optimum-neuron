# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.

import hashlib
import importlib
import json
import shlex
from functools import wraps
from typing import Any, Callable

from ..utils.patching import patch_everywhere


_NXD_TRACE_MODULES = (
    "neuronx_distributed.trace.model_builder",
    "neuronx_distributed.trace.model_builder_utils",
    "neuronx_distributed.trace.functions",
    "neuronx_distributed.trace.hlo_utils",
    "neuronx_distributed.trace.model_builder_v2",
)


def _clear_field(message: Any, field_name: str) -> None:
    descriptor = getattr(message, "DESCRIPTOR", None)
    if descriptor is not None and field_name in descriptor.fields_by_name:
        message.ClearField(field_name)


def _normalize_instruction(
    instruction: Any, instruction_id_map: dict[int, int], computation_id_map: dict[int, int]
) -> None:
    original_instruction_id = instruction.id
    instruction.id = instruction_id_map.get(original_instruction_id, original_instruction_id)
    instruction.name = f"instruction_{instruction.id}"

    if hasattr(instruction, "operand_ids"):
        instruction.operand_ids[:] = [
            instruction_id_map.get(operand_id, operand_id) for operand_id in instruction.operand_ids
        ]
    if hasattr(instruction, "control_predecessor_ids"):
        instruction.control_predecessor_ids[:] = [
            instruction_id_map.get(predecessor_id, predecessor_id)
            for predecessor_id in instruction.control_predecessor_ids
        ]
    if hasattr(instruction, "called_computation_ids"):
        instruction.called_computation_ids[:] = [
            computation_id_map.get(computation_id, computation_id)
            for computation_id in instruction.called_computation_ids
        ]

    if hasattr(instruction, "metadata"):
        instruction.metadata.Clear()
    if hasattr(instruction, "frontend_attributes"):
        instruction.frontend_attributes.Clear()


def _normalize_schedule(
    hlo_module: Any, computation_id_map: dict[int, int], instruction_id_maps: dict[int, dict[int, int]]
) -> None:
    if not hasattr(hlo_module, "schedule"):
        return

    for sequence in hlo_module.schedule.sequences:
        original_computation_id = sequence.key
        sequence.key = computation_id_map.get(original_computation_id, original_computation_id)

        instruction_id_map = instruction_id_maps.get(original_computation_id, {})
        sequence.value.instruction_ids[:] = [
            instruction_id_map.get(instruction_id, instruction_id) for instruction_id in sequence.value.instruction_ids
        ]


def canonicalize_hlo_module(hlo_module: Any) -> Any:
    canonical_hlo = type(hlo_module)()
    canonical_hlo.CopyFrom(hlo_module)

    original_entry_computation_id = getattr(canonical_hlo, "entry_computation_id", None)
    original_entry_computation_name = getattr(canonical_hlo, "entry_computation_name", None)

    _clear_field(canonical_hlo, "id")
    _clear_field(canonical_hlo, "name")
    _clear_field(canonical_hlo, "profile_info")
    _clear_field(canonical_hlo, "stack_frame_index")
    if hasattr(canonical_hlo, "frontend_attributes"):
        canonical_hlo.frontend_attributes.Clear()

    computation_id_map = {}
    for computation_index, computation in enumerate(canonical_hlo.computations, start=1):
        computation_id_map[computation.id] = computation_index

    instruction_id_maps = {}
    for computation_index, computation in enumerate(canonical_hlo.computations, start=1):
        original_computation_id = computation.id
        original_root_id = computation.root_id

        instruction_id_map = {
            instruction.id: instruction_index
            for instruction_index, instruction in enumerate(computation.instructions, start=1)
        }
        instruction_id_maps[original_computation_id] = instruction_id_map

        computation.id = computation_id_map.get(original_computation_id, computation_index)
        computation.name = (
            "entry"
            if original_computation_id == original_entry_computation_id
            or computation.name == original_entry_computation_name
            else f"computation_{computation.id}"
        )
        computation.root_id = instruction_id_map.get(original_root_id, original_root_id)
        _clear_field(computation, "execution_thread")

        for instruction in computation.instructions:
            _normalize_instruction(instruction, instruction_id_map, computation_id_map)

    if original_entry_computation_id is not None:
        canonical_hlo.entry_computation_id = computation_id_map.get(
            original_entry_computation_id, original_entry_computation_id
        )
    if canonical_hlo.computations:
        canonical_hlo.entry_computation_name = "entry"

    _normalize_schedule(canonical_hlo, computation_id_map, instruction_id_maps)
    return canonical_hlo


def serialize_canonical_hlo_module(hlo_module: Any) -> bytes:
    canonical_hlo = canonicalize_hlo_module(hlo_module)
    return canonical_hlo.SerializeToString(deterministic=True)


def _canonical_flags_text(flags: str | list[str] | tuple[str, ...] | None) -> str:
    canonical_flags = canonicalize_compiler_flags(flags)
    return "".join(canonical_flags)


def canonicalize_compiler_flags(flags: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if flags is None:
        return []
    if isinstance(flags, str):
        tokens = shlex.split(flags)
    else:
        tokens = list(flags)

    canonical_flags = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "--logfile":
            skip_next = True
            continue
        if token.startswith("--logfile="):
            continue
        canonical_flags.append(token)
    return canonical_flags


def canonical_hlo_hash(
    hlo_module: Any, flags: str | list[str] | tuple[str, ...] | None = None, prefix_len: int = 20
) -> str:
    hash_gen = hashlib.sha256()
    hash_gen.update(serialize_canonical_hlo_module(hlo_module))
    canonical_flags = _canonical_flags_text(flags)
    if canonical_flags:
        hash_gen.update(canonical_flags.encode("utf-8"))
    return hash_gen.hexdigest()[:prefix_len]


def canonical_get_hash_module(hlo_module: Any, flags: str | None) -> str:
    return canonical_hlo_hash(hlo_module, flags=flags, prefix_len=20)


def canonical_generate_key(hlo_module: Any, key: str | None = None) -> str:
    if key is not None:
        return key
    return f"model_{canonical_hlo_hash(hlo_module, prefix_len=8)}"


def canonical_get_compiler_flags_hash(compiler_flags: str | list[str] | tuple[str, ...] | None) -> str:
    compile_flags_str = json.dumps(canonicalize_compiler_flags(compiler_flags))
    return hashlib.md5(compile_flags_str.encode()).hexdigest()[:8]


def _canonicalize_cache_key_from_hlo_file(
    input_file: str, compiler_flags: Any, fallback_cache_key: str | None
) -> str | None:
    try:
        from libneuronxla.proto import hlo_pb2

        module = hlo_pb2.HloModuleProto()
        with open(input_file, "rb") as handle:
            module.ParseFromString(handle.read())
        return canonical_hlo_hash(module, flags=compiler_flags, prefix_len=20)
    except Exception:
        return fallback_cache_key


def _wrap_neuron_xla_compile_impl(compile_impl: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(compile_impl)
    def wrapper(input_file: str, compiler_flags: Any, output: str, *args: Any, **kwargs: Any) -> Any:
        kwargs["cache_key"] = _canonicalize_cache_key_from_hlo_file(
            input_file=input_file,
            compiler_flags=compiler_flags,
            fallback_cache_key=kwargs.get("cache_key"),
        )
        canonical_flags = canonicalize_compiler_flags(compiler_flags)
        return compile_impl(input_file, canonical_flags, output, *args, **kwargs)

    return wrapper


def patch_cache_key_canonicalization() -> Callable[[], None]:
    imported_modules = {}
    for module_name in _NXD_TRACE_MODULES:
        try:
            imported_modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            continue

    neuron_cc_cache = None
    neuron_cc_wrapper = None
    try:
        neuron_cc_cache = importlib.import_module("libneuronxla.neuron_cc_cache")
    except ImportError:
        neuron_cc_cache = None
    try:
        neuron_cc_wrapper = importlib.import_module("libneuronxla.neuron_cc_wrapper")
    except ImportError:
        neuron_cc_wrapper = None

    if not imported_modules and neuron_cc_cache is None and neuron_cc_wrapper is None:
        return lambda: None

    original_generate_key = None
    original_get_hash_module = None
    original_get_compiler_flags_hash = None
    original_neuron_xla_compile_impl = None

    for module in imported_modules.values():
        if original_generate_key is None and hasattr(module, "generate_key"):
            original_generate_key = getattr(module, "generate_key")
        if original_get_hash_module is None and hasattr(module, "get_hash_module"):
            original_get_hash_module = getattr(module, "get_hash_module")

    if neuron_cc_cache is not None:
        original_get_compiler_flags_hash = neuron_cc_cache.CompileCache.get_compiler_flags_hash
        neuron_cc_cache.CompileCache.get_compiler_flags_hash = staticmethod(canonical_get_compiler_flags_hash)

    if neuron_cc_wrapper is not None:
        original_neuron_xla_compile_impl = neuron_cc_wrapper.neuron_xla_compile_impl
        neuron_cc_wrapper.neuron_xla_compile_impl = _wrap_neuron_xla_compile_impl(
            neuron_cc_wrapper.neuron_xla_compile_impl
        )

    patch_everywhere("generate_key", canonical_generate_key, "neuronx_distributed.trace")
    patch_everywhere("get_hash_module", canonical_get_hash_module, "neuronx_distributed.trace")

    def restore() -> None:
        if original_generate_key is not None:
            patch_everywhere("generate_key", original_generate_key, "neuronx_distributed.trace")
        if original_get_hash_module is not None:
            patch_everywhere("get_hash_module", original_get_hash_module, "neuronx_distributed.trace")
        if neuron_cc_cache is not None and original_get_compiler_flags_hash is not None:
            neuron_cc_cache.CompileCache.get_compiler_flags_hash = original_get_compiler_flags_hash
        if neuron_cc_wrapper is not None and original_neuron_xla_compile_impl is not None:
            neuron_cc_wrapper.neuron_xla_compile_impl = original_neuron_xla_compile_impl

    return restore
