# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import hashlib
import json
import logging
import math
import os
import shlex
import shutil
import subprocess
from contextlib import contextmanager, nullcontext
from textwrap import dedent

import numpy as np
import torch
from libneuronxla import neuron_xla_compile
from libneuronxla.neuron_cc_cache import CacheUrl, create_compile_cache
from neuronxcc import __version__ as compiler_version
from torch_neuronx.proto import metaneff_pb2
from torch_neuronx.pyhlo import xla_data_pb2
from torch_neuronx.pyhlo.constant.serialize_torch import serialize_torch
from torch_neuronx.pyhlo.scribe import HloScribe

from . import ops


def get_hash_module(hlo_module, flags):
    # Hashing is pretty fast and neglegible compared to compilation time
    hash_gen = hashlib.sha256()
    text = str(hlo_module)
    if flags is not None:
        text += flags.replace(" ", "")
    hash_gen.update(text.encode("utf-8"))
    hash = str(hash_gen.hexdigest())[:20]
    return hash


@contextmanager
def envvar(key, value):
    prior = os.environ.pop(key, None)
    if value is not None:
        os.environ[key] = value
    try:
        yield
    finally:
        os.environ.pop(key, None)
        if prior is not None:
            os.environ[key] = prior


def compile_py_func(py_func):
    # Adds file/scope metadata during debug dump
    context = nullcontext()
    if "NEURONX_DUMP_TO" in os.environ and "ENABLE_PYHLO_FILE_METADATA" not in os.environ:
        context = envvar("ENABLE_PYHLO_FILE_METADATA", "1")

    with context:
        return HloScribe(serialize_torch)(py_func).module_proto


def get_compiler_flags() -> str:
    user_flags = os.environ.get("NEURON_CC_FLAGS", "").strip()

    default_flags = {
        "--model-type": "transformer",
        "--auto-cast": "none",
    }

    # Set default flag where there is no user-provided value
    flags = [user_flags] if user_flags else []
    for key, value in default_flags.items():
        if key not in user_flags:
            flags.append(f"{key}={value}")

    # Reformat list of flags to joined string
    return " ".join(flags)


def compile_hlo_module(hlo_module, tag=None, num_exec_repetition=1):
    flags = get_compiler_flags()
    flags = f"{flags} --execute-repetition={num_exec_repetition}"

    module_flag_hash = get_hash_module(hlo_module, flags)
    module_hash = get_hash_module(hlo_module, None)

    dump = "NEURONX_DUMP_TO" in os.environ
    neff_bytes = None

    # tag is used to make folder name more clear (e.g. add bucket-size to folder name)
    if tag is None:
        hlo_module_name = f"{hlo_module.name}.{compiler_version}.{module_flag_hash}"
    else:
        hlo_module_name = f"{tag}-{hlo_module.name}.{compiler_version}.{module_flag_hash}"

    if dump:
        dump_to_parent = os.environ.get("NEURONX_DUMP_TO", "/tmp")
        dump_to = os.path.join(dump_to_parent, hlo_module_name)
        os.makedirs(dump_to, exist_ok=True)
        hlo_module_path = os.path.join(dump_to, f"{hlo_module_name}.pb")
        hlo_module_path = os.path.realpath(hlo_module_path)
        if not os.path.isfile(hlo_module_path):
            dump_proto(hlo_module, hlo_module_path)
        neff_path = f"{hlo_module_path}.neff"
        neff_path = os.path.realpath(neff_path)
        if not os.path.exists(neff_path):
            command_line = [
                "neuronx-cc",
                "compile",
                "--framework=XLA",
                "--target=trn1",
                hlo_module_path,
                f"--output={neff_path}",
                *shlex.split(flags),
            ]
            command_line.extend(["--verbose=INFO", "--pipeline", "compile", "SaveTemps"])
            subprocess.check_call(command_line, cwd=dump_to)
        with open(neff_path, "rb") as f:
            neff_bytes = f.read()
        try:
            shutil.copyfile(
                os.path.join(dump_to_parent, "neuron_model_config.json"),
                os.path.join(dump_to, "neuron_model_config.json"),
            )
        except FileNotFoundError:
            pass
    else:
        module_bytes = hlo_module.SerializeToString()
        try:
            neff_bytes = neuron_xla_compile(
                module_bytes,
                flags,
                input_format="hlo",
                platform_target="trn1",
                cache_key=module_hash,
                retry_failed_compilation=False,
                lazy=True,
                use_cache=True,
                cache_dir=None,
            )
        finally:
            notemp_dump = os.environ.get("NEURONX_DUMP_TO_NOTEMP", None)

            # for torch 2.0 pjrt compatibility, check if download_artifacts is implemented
            cache_url = CacheUrl.get_cache_url()
            compile_cache = create_compile_cache(cache_url)
            if notemp_dump and hasattr(compile_cache, "download_artifacts"):
                dump_path = os.path.join(notemp_dump, hlo_module_name)
                os.makedirs(dump_path, exist_ok=True)

                parent_dir, cache_key = compile_cache.get_cache_dir(module_hash, flags)
                compile_cache.download_artifacts(parent_dir, dump_path)

    return neff_bytes


def dump_proto(proto, path):
    with open(path, "wb") as f:
        f.write(proto.SerializeToString())


def hlo2metaneff(hlo_module):
    prog_shape = hlo_module.host_program_shape
    dtype_converter = DataTypeConverter()

    def fill_with(target, names, shapes):
        for name, shape in zip(names, shapes):
            tensor = target.add()
            tensor.name = name.encode()
            tensor.shape[:] = shape.dimensions
            tensor.data_type = dtype_converter.hlo2metaneff(shape.element_type)

    input_names = find_input_names(hlo_module)
    metaneff = metaneff_pb2.MetaNeff()
    fill_with(metaneff.input_tensors, input_names, prog_shape.parameters)
    output_names = find_output_names(hlo_module)
    if prog_shape.result.element_type == xla_data_pb2.PrimitiveType.TUPLE:
        output_shapes = prog_shape.result.tuple_shapes
    else:
        output_shapes = [prog_shape.result]
    fill_with(metaneff.output_tensors, output_names, output_shapes)
    for entry in hlo_module.input_output_alias.entries:
        assert len(entry.parameter_shape_index) == 0
        metaneff.output_aliases_to[entry.output_shape_index[0]] = entry.parameter_number
    return metaneff


def find_input_names(hlo_module):
    # TODO: read names from hlo_module
    prog_shape = hlo_module.host_program_shape
    return [f"input{idx}" for idx in range(len(prog_shape.parameters))]


def find_output_names(hlo_module):
    # TODO: read names from hlo_module
    prog_shape = hlo_module.host_program_shape
    if prog_shape.result.element_type != xla_data_pb2.PrimitiveType.TUPLE:
        return ["output0"]
    return [f"output{idx}" for idx in range(len(prog_shape.result.tuple_shapes))]


class DataTypeConverter:
    def __init__(self):
        name_mapping = """
            PRED    UINT8       bool
            S8      INT8        int8
            S16     INT16       int16
            S32     INT32       int32
            S64     INT64       int64
            U8      UINT8       uint8
            U16     UINT16      int16
            U32     INT32       int32
            U64     INT64       int64
            F16     FLOAT16     float16
            F32     FLOAT       float32
            F64     DOUBLE      float64
            BF16    BFLOAT16    bfloat16
            F8E4M3FN INT8     float8_e4m3fn
        """
        # Note that for FP8 we map metaneff datatype to int8, since from the runtime perspective these datatypes are functionally equivalent (for fp8 storage only)
        # Within Tnx, we no longer use the metaneff flow, so this would not matter anyway.
        name_mapping = dedent(name_mapping)
        name_mapping = name_mapping.lstrip().strip()
        self.hlo2metaneff_mapping = {}
        self.hlo2torch_mapping = {}
        self.torch2name_mapping = {}
        self.torch2hlo_mapping = {}
        for line in name_mapping.split("\n"):
            line = line.lstrip().strip()
            pname, dname, tname = line.split()
            if not hasattr(torch, tname):
                continue
            primitive_type = getattr(xla_data_pb2.PrimitiveType, pname)
            metaneff_dtype = getattr(metaneff_pb2.MetaTensor.DataType, dname)
            torch_dtype = getattr(torch, tname)
            self.hlo2metaneff_mapping[primitive_type] = metaneff_dtype
            self.hlo2torch_mapping[primitive_type] = torch_dtype
            self.torch2name_mapping[torch_dtype] = pname.lower()
            self.torch2hlo_mapping[torch_dtype] = primitive_type

    def hlo2metaneff(self, primitive_type):
        return self.hlo2metaneff_mapping[primitive_type]

    def hlo2torch(self, primitive_type):
        return self.hlo2torch_mapping[primitive_type]

    def torch2name(self, torch_dtype):
        return self.torch2name_mapping[torch_dtype]

    def torch2hlo(self, torch_dtype):
        return self.torch2hlo_mapping[torch_dtype]


class ParallelMemory:
    def __init__(self, hlo_module, tp_degree):
        input_names = find_input_names(hlo_module)
        output_names = find_output_names(hlo_module)
        self.inputs = torch.classes.neuron.ParallelTensorSet(input_names, tp_degree)
        self.outputs = torch.classes.neuron.ParallelTensorSet(output_names, tp_degree)
        self.input_tensors = None
        self.output_tensors = None
        self.n_debug_tensors = 0  # How many of output tensors are for debugging

    def init(self):
        self.inputs.init()
        self.outputs.init()

    def setup(self, input_tensors, output_tensors, n_debug_tensors=0):
        self.n_debug_tensors = n_debug_tensors
        self.inputs.init()
        self.outputs.init()
        for idx, tensor in enumerate(input_tensors):
            self.inputs.add(idx, tensor)
        for idx, tensor in enumerate(output_tensors):
            self.outputs.add(idx, tensor)
        self.input_tensors = input_tensors
        self.output_tensors = output_tensors


class Executor:
    def __init__(self, kernel, memory, inputs, outputs):
        """
        An optimized executor class that allocates a thread for each model rank.

        This implements a number of optimizations to speed up execution time:
        - The primary optimization is that this class creates set of fixed
          threads that are always blocking on a barrier waiting to execute.
          In practice, this can significantly speed up the thread startup
          time for model executions compared to other threadpool
          implementations.
        - A second optimization is that each thread is responsible for
          both the I/O and the execution of the model rank. This means that
          fewer `libtorchneuron` API calls are made an fewer threadpool starts
          are being triggered.
        The combination of the above optimization can have a huge effect
        especially on small/fast models.
        """
        self.kernel = kernel
        self.memory = memory
        self.inputs = inputs
        self.outputs = outputs
        self.executor = torch.classes.neuron.ParallelExecutor(
            kernel.model,
            memory.inputs,  # All inputs (inputs, caches, weights)
            memory.outputs,  # All outputs (outputs, caches)
            inputs,  # User provided inputs
            outputs,  # Returned outputs
        )

    def __call__(self, inputs, return_ranks: int = -1):
        """
        Execute the kernel with the given inputs.

        Arguments:
            inputs: A set of input tensors to copy to each model rank.
            return_ranks: Specifies which ranks to return to python. This is
                useful if data is only required from a specific number of
                NeuronCores. For example:
                    * 0: Do not return any data.
                    * 1: Return data only from the first rank. This is useful
                         when the data is synchronized across ranks.
                    * -1: Return data form all ranks.

        Returns:
            result: The output tensors from each rank concatenated along dim 0.
        """
        casted = []
        for i, (cpu, buf) in enumerate(zip(inputs, self.inputs)):
            if cpu.shape != buf.shape:
                raise AssertionError(f"{i + 1}th input shape mismatch. Expected {buf.shape}, but got {cpu.shape}")
            if cpu.dtype != buf.dtype:
                cpu = cpu.to(buf.dtype)
            casted.append(cpu)

        if self.kernel.snapshot is not None:
            if self.kernel.snapshot_steps is None or ParallelKernel.hlo_snapshot_iter in self.kernel.snapshot_steps:
                self.kernel.snapshot_enter(self.memory.input_tensors)
                self.kernel.snapshot_tensors(inputs, "inputs")  # Overwrite with current values
                outputs = torch.ops.neuron._parallel_executor_run(self.executor, casted, return_ranks)
                self.kernel.snapshot_exit(self.memory.output_tensors)
            else:
                outputs = torch.ops.neuron._parallel_executor_run(self.executor, casted, return_ranks)
                ParallelKernel.hlo_snapshot_iter += 1
        else:
            outputs = torch.ops.neuron._parallel_executor_run(self.executor, casted, return_ranks)

        # Executor v1 returns multiple shards
        # Executor v2 returns concatenated shards (across dim=0) in one contiguous allocation
        # This logic is compatible with both new/old versions of `libtorchneuron.so`
        if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            result = tuple(outputs)
        elif return_ranks == 1:
            result = tuple(shards[0] for shards in outputs)
        else:
            result = tuple(torch.cat(shards, dim=0) for shards in outputs)
        if len(result) == 1:
            return result[0]
        return result


def write_tensors(tensors, folder, worker=0):
    os.makedirs(folder, exist_ok=True)
    for i, tensor in enumerate(tensors):
        filename = os.path.join(folder, f"{i}.npy")
        if tensor.device != torch.device("cpu"):
            tensor = ops.parallel_cpu(tensor)
            if isinstance(tensor, list):
                tensor = tensor[worker]
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.view(torch.int16)
            tensor = tensor.numpy()
            tensor = tensor.view("|V2")
        else:
            tensor = tensor.detach().numpy()
        np.save(filename, tensor)


@contextmanager
def io_ring_cache_context(size):
    """
    A context which temporarily sets the IO ring cache size if it is not set.

    The IO ring cache size environment variable controls the number of tensor
    descriptor cache slots which are allocated to NeuronCore memory. This
    tensor information is required to execute a NEFF. If the cache is not
    configured, the Neuron runtime will generate tensor memory information upon
    every execution. Caching improves performance by reusing the tensor
    information after it has been generated the first time.

    The number of allocated IO ring slots can be changed individually for
    each NEFF by changing the environment variable prior to the call to load.

    For optimal performance, this cache should be configured to be equal to the
    number *unique* sets of weights used per NEFF:
    - For a fully unrolled network the cache size should be 1 since it has
      exactly 1 set of weights (all weights for all layers).
    - For a multi-layer network (partial unroll), the cache size should equal
      `n_layers / unroll` since the neff will be be executed that many times
      with a different set of unique weights.

    Arguments:
        size: The number of cache slots to allocate for IO descriptors.
    """
    key = "NEURON_RT_IO_RING_CACHE_SIZE"
    if os.environ.get(key, None) is not None:
        yield  # Do nothing if we have a user-provided cache configuration
    else:
        os.environ[key] = str(size)
        yield
        os.environ.pop(key)


class ParallelKernel:
    hlo_snapshot_iter = 0

    def __init__(
        self,
        hlo_module,
        tp_degree,
        g_start_device_id=0,
        g_device_count=None,
        tag=None,
        num_exec_repetition=1,
    ):
        self.hlo_module = hlo_module
        self.tp_degree = tp_degree
        self.neff_bytes = None
        self.model = None
        self.snapshot = os.environ.get("HLO_SNAPSHOT_PATH", None)
        self.snapshot_steps = os.environ.get("HLO_SNAPSHOT_STEPS", None)
        if self.snapshot_steps:
            self.snapshot_steps = json.loads(self.snapshot_steps)
        self.g_start_device_id = g_start_device_id
        if g_device_count is None:
            g_device_count = tp_degree
        self.tag = tag
        self.g_device_count = g_device_count
        self.memories = []
        self.num_exec_repetition = num_exec_repetition
        self.total_input_tensors_size = get_total_input_tensors_size(self.hlo_module)
        logging.debug(
            f"Total input tensor size of the module (per rank): {self.total_input_tensors_size / (10**9)} G, whole (all ranks): {self.total_input_tensors_size * tp_degree / (10**9)} G"
        )

    def build_memory(self):
        memory = ParallelMemory(self.hlo_module, self.tp_degree)
        self.memories.append(memory)
        return memory

    def compile(self, num_exec_repetition=1):
        self.build(num_exec_repetition)
        return self.neff_bytes

    def build(self, num_exec_repetition=1):
        # Avoid rebuilding NEFF. This path occurs during deserialization
        if self.neff_bytes is not None:
            return
        self.neff_bytes = compile_hlo_module(self.hlo_module, self.tag, num_exec_repetition)

    def load(self, io_ring_cache_size=1):
        assert self.neff_bytes is not None, "Try to load with neff bytes as None, might due to compilation failure"
        self.model = torch.classes.neuron.ParallelModel(
            self.neff_bytes, self.tp_degree, self.g_start_device_id, self.g_device_count
        )
        with io_ring_cache_context(io_ring_cache_size):
            logging.debug(
                f"loading model with tp_degree {self.tp_degree}, g_start_device_id {self.g_start_device_id} g_device_count {self.g_device_count}"
            )
            self.model.load()

    def warmup(self):
        """
        Execute the kernel with each memory once.

        This ensure that any initialization latency related to caching or
        runtime setup is already complete prior to the model being executed.
        """
        # Note: Explicitly turn off snapshot during warmup
        snapshot = self.snapshot
        self.snapshot = None
        for memory in self.memories:
            try:
                self(memory)
            except Exception:
                # Ignoring exceptions avoids uninitialized memory related errors
                pass
        self.snapshot = snapshot

    def snapshot_path(self):
        suffix = ""
        if self.tag is not None:
            suffix = f"-{self.tag}"
        path = os.path.join(self.snapshot, f"iter{ParallelKernel.hlo_snapshot_iter}{suffix}")
        os.makedirs(path, exist_ok=True)
        return path

    def snapshot_tensors(self, inputs, subdir):
        folder = self.snapshot_path()
        snapshot_all_workers = os.environ.get("HLO_SNAPSHOT_ALL_WORKERS", None)
        if snapshot_all_workers:
            for worker in range(self.tp_degree):
                path = os.path.join(folder, subdir, f"worker{worker}")
                write_tensors(inputs, path, worker)
        else:
            path = os.path.join(folder, subdir)
            write_tensors(inputs, path)

    def snapshot_enter(self, inputs):
        folder = self.snapshot_path()
        path = os.path.join(folder, "graph.hlo.pb")
        with open(path, "wb") as f:
            f.write(self.hlo_module.SerializeToString())
        path = os.path.join(folder, "graph.neff")
        with open(path, "wb") as f:
            f.write(self.neff_bytes)
        self.snapshot_tensors(inputs, "inputs")

    def snapshot_exit(self, outputs):
        self.snapshot_tensors(outputs, "outputs")
        ParallelKernel.hlo_snapshot_iter += 1

    def __call__(self, memory):
        logging.debug(f"running {self.hlo_module.name}")
        if self.snapshot is not None:
            if self.snapshot_steps is None or ParallelKernel.hlo_snapshot_iter in self.snapshot_steps:
                self.snapshot_enter(memory.input_tensors)
                result = ops.parallel_run(self.model, memory.inputs, memory.outputs)
                self.snapshot_exit(memory.output_tensors)
                return result
            else:
                ParallelKernel.hlo_snapshot_iter += 1
        return ops.parallel_run(self.model, memory.inputs, memory.outputs)

    def build_executor(self, memory, inputs, outputs):
        return Executor(self, memory, inputs, outputs)

    def profile(self, profile_dir, ntff_count_limit):
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir, exist_ok=True)

        tagged_hlo = (f"{self.tag}-" if self.tag else "") + self.hlo_module.name
        # Replace problem characters used in filenames
        tagged_hlo = tagged_hlo.translate(str.maketrans("(), ", "__x_"))
        ntff_prefix = os.path.join(profile_dir, tagged_hlo)

        # Set up inputs as zeros
        for t in self.memories[0].input_tensors:
            zero = torch.zeros(t.shape, dtype=t.dtype)
            zeros = [zero] * self.tp_degree
            ops.parallel_write(t, zeros)

        # Warm up inference
        self(self.memories[0])

        # Profile start numbered NTFF files f"{ntff_prefix}_rank_0.ntff" etc
        # Allow user to limit the number of NTFF files generated
        self.ntff_paths = ops.parallel_profile_start(self.model, ntff_prefix, ntff_count_limit)

        # Single inference in the profile loop
        self(self.memories[0])

        # Profile stop
        ops.parallel_profile_stop(self.ntff_paths)

        # Save NEFF file
        neff_filename = os.path.join(profile_dir, f"{tagged_hlo}.neff")
        with open(neff_filename, "wb") as f:
            f.write(self.neff_bytes)

        # Tar file was part of th original code, comment for now to
        # save disk space
        """
        ntff_tar_path = os.path.join(profile_dir,
                                     f'{tagged_hlo}.profile.tar')
        with tarfile.open(ntff_tar_path, 'w|') as fp:
            fp.add(neff_filename)
            for ntff_path in self.ntff_paths:
                fp.add(ntff_path)
        """


def gen_zero_input(hlo_module, index):
    shape_proto = hlo_module.host_program_shape.parameters[index]
    shape = list(shape_proto.dimensions)
    dtype = DataTypeConverter().hlo2torch(shape_proto.element_type)
    return torch.zeros(shape, dtype=dtype)


def gen_zero_output(hlo_module, index=None):
    shape_proto = hlo_module.host_program_shape.result
    if index is not None:
        shape_proto = shape_proto.tuple_shapes[index]
    shape = list(shape_proto.dimensions)
    dtype = DataTypeConverter().hlo2torch(shape_proto.element_type)
    return torch.zeros(shape, dtype=dtype)


def get_total_input_tensors_size(hlo_module):
    total_bytes = 0
    dtype_converter = DataTypeConverter()
    for _, param in enumerate(hlo_module.host_program_shape.parameters):
        dtype = dtype_converter.hlo2torch(param.element_type)
        if dtype.is_floating_point:
            num_bytes = math.prod(param.dimensions) * torch.finfo(dtype).bits / 8
        else:
            num_bytes = math.prod(param.dimensions) * torch.iinfo(dtype).bits / 8
        total_bytes += num_bytes
    return total_bytes
