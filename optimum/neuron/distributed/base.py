# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Base class related to `neuronx_distributed` to perform parallelism."""

import contextlib
import gc
import math
from abc import ABC, abstractclassmethod
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type, Union

import torch
from transformers import PreTrainedModel

from ...utils import logging
from ..utils import is_neuronx_distributed_available, is_torch_xla_available
from ..utils.misc import is_main_worker, is_precompilation
from ..utils.model_utils import (
    get_parent_module_and_param_name_from_fully_qualified_name,
    get_tied_parameters_dict,
)
from ..utils.patching import Patcher
from ..utils.peft_utils import NeuronPeftModel
from ..utils.require_utils import requires_neuronx_distributed, requires_torch_xla
from .parallel_layers import (
    IOSequenceParallelizer,
    LayerNormSequenceParallelizer,
    LayerNormType,
    SequenceCollectiveOpInfo,
)
from .utils import (
    MODEL_PARALLEL_SHARDS_DIR_NAME,
    OptimumGQAQKVColumnParallelLinear,
    OptimumNeuronFXTracer,
    WeightInformation,
    get_base_model_and_peft_prefix,
    get_linear_weight_info,
    get_output_projection_qualified_names_after_qga_qkv_replacement,
    get_parameter_names_mapping_after_gqa_qkv_replacement,
    get_parameters_tp_metadata,
    initialize_parallel_linear,
    initialize_torch_nn_module,
    linear_to_parallel_linear,
    load_tensor_for_weight,
    maybe_load_linear_weight_to_gqa_qkv_column_parallel_linear,
    maybe_load_linear_weight_to_parallel_linear,
    maybe_load_weights_to_gqa_qkv_column_parallel_linear,
    maybe_load_weights_to_output_projection_when_using_gqa_qkv_column_parallel_linear,
    parameter_can_be_initialized,
    try_to_hf_initialize,
    was_already_initialized_during_parallelization,
)


if TYPE_CHECKING:
    if is_neuronx_distributed_available():
        from neuronx_distributed.pipeline import NxDPPModel

logger = logging.get_logger()


class SequenceParallelismSpecs:
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS: Optional[List[str]] = None
    LAYERNORM_TYPE: LayerNormType = LayerNormType.REGULAR
    SEQUENCE_COLLECTIVE_OPS_INFOS: Optional[List[SequenceCollectiveOpInfo]] = None

    @abstractclassmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        """
        This method needs to be overriden. It must patch anything model-specfic to make the model compatible with
        sequence parallelism.
        """
        if sequence_parallel_enabled:
            raise NotImplementedError(
                f"No patching for the attention mechanism for sequence parallelism was implemented for {model.__class__}"
            )


class PipelineParallelismSpecs:
    TRASNFORMER_LAYER_CLS: Type[torch.nn.Module]
    DEFAULT_INPUT_NAMES: Union[Tuple[str, ...], Dict[str, Tuple[str, ...]]]
    LEAF_MODULE_CLASSES_NAMES: Optional[List[Union[str, Type[torch.nn.Module]]]] = None
    OUTPUT_LOSS_SPECS: Tuple[bool, ...] = (True, False)

    @classmethod
    @requires_torch_xla
    def create_pipeline_cuts(cls, model: PreTrainedModel, pipeline_parallel_size: int, log: bool = True) -> List[str]:
        """
        Creates the pipeline cuts, e.g. the name of the layers at each the cuts happen for pipeline parallelism.
        """

        num_layers = sum(1 if isinstance(mod, cls.TRASNFORMER_LAYER_CLS) else 0 for mod in model.modules())
        if num_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"The number of transformer layers ({num_layers}) is not divisible by the pipeline parallel size "
                f"({pipeline_parallel_size})."
            )
        num_layers_per_partition = num_layers // pipeline_parallel_size
        layers_names = [name for (name, mod) in model.named_modules() if isinstance(mod, cls.TRASNFORMER_LAYER_CLS)]
        pipeline_cuts = [
            layers_names[cut_idx]
            for cut_idx in range(num_layers_per_partition - 1, num_layers - 1, num_layers_per_partition)
        ]

        if is_main_worker() and log:
            logger.info(f"Pipeline parallelism cuts: {pipeline_cuts}.")

        return pipeline_cuts

    @classmethod
    def leaf_module_cls(cls) -> List[str]:
        if cls.LEAF_MODULE_CLASSES_NAMES is None:
            return []
        return [class_ if isinstance(class_, str) else class_.__name__ for class_ in cls.LEAF_MODULE_CLASSES_NAMES]

    @classmethod
    def get_patching_specs(cls) -> List[Tuple[str, Any]]:
        return []


class Parallelizer(ABC):
    """
    Base abstract class that handles model parallelism.
    """

    SEQUENCE_PARALLELSIM_SPECS_CLS: Optional[Type[SequenceParallelismSpecs]] = None
    PIPELINE_PARALLELISM_SPECS_CLS: Optional[Type[PipelineParallelismSpecs]] = None

    def __init__(self):
        self._validate_required_libaries_are_available()

    def _validate_required_libaries_are_available(self):
        if not is_neuronx_distributed_available():
            raise RuntimeError(
                "Parallelizer requires the `neuronx_distributed` package. You can install it by running: pip install "
                "neuronx_distributed"
            )
        if not is_torch_xla_available():
            raise RuntimeError(
                "Parallelizer requires the `torch_xla` package. You can install it by running: pip install torch_xla"
            )

    @classmethod
    @contextlib.contextmanager
    def saved_model_in_temporary_directory(cls, model: "PreTrainedModel"):
        tmpdir = TemporaryDirectory()
        path = Path(tmpdir.name) / "pytorch_model.bin"
        torch.save({"model": model.state_dict()}, path.as_posix())
        try:
            yield path
        finally:
            tmpdir.cleanup()

    @classmethod
    def supports_sequence_parallelism(cls) -> bool:
        return cls.SEQUENCE_PARALLELSIM_SPECS_CLS is not None

    @classmethod
    def supports_pipeline_parallelism(cls) -> bool:
        return cls.PIPELINE_PARALLELISM_SPECS_CLS is not None

    @classmethod
    @requires_neuronx_distributed
    def _get_parameter_names_for_current_pipeline(
        cls, model: torch.nn.Module, remove_duplicate: bool = True
    ) -> Set[str]:
        """
        Retrieves the names of the parameters that will be in the current pipeline stage by using the pipeline
        parallelism rank.
        """
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_pipeline_model_parallel_rank,
            get_pipeline_model_parallel_size,
        )

        pp_size = get_pipeline_model_parallel_size()
        pp_rank = get_pipeline_model_parallel_rank()
        all_parameter_names = {n for n, _ in model.named_parameters(remove_duplicate=remove_duplicate)}
        if pp_size == 1:
            return all_parameter_names

        if not cls.supports_pipeline_parallelism():
            raise NotImplementedError(f"{cls} does not support pipeline parallelism.")

        cuts = cls.PIPELINE_PARALLELISM_SPECS_CLS.create_pipeline_cuts(model, pp_size, log=False)

        start_module_name = cuts[pp_rank - 1] if pp_rank >= 1 else None
        end_module_name = None if pp_rank == pp_size - 1 else cuts[pp_rank]
        parameter2name = {p: n for n, p in model.named_parameters(remove_duplicate=remove_duplicate)}
        parameter_names = set()
        should_add = False
        for name, mod in model.named_modules():
            if not isinstance(mod, cls.PIPELINE_PARALLELISM_SPECS_CLS.TRASNFORMER_LAYER_CLS):
                continue
            # If start_module_name is None, it means we are on the first rank, we should add right from the beginning.
            if start_module_name is None:
                should_add = True
            if should_add:
                for _, param in mod.named_parameters(remove_duplicate=remove_duplicate):
                    # It is important to use this dictionary (built with `model.named_parameters()`) instead of using
                    # `mod.named_parameters()` to get the fully qualified names.
                    param_name = parameter2name[param]
                    parameter_names.add(param_name)

            # We consider the parameters inside ]start_module_name, end_module_name].
            if start_module_name == name:
                should_add = True
            if name == end_module_name:
                break

        parameters_inside_transformer_layers = {
            p
            for mod in model.modules()
            if isinstance(mod, cls.PIPELINE_PARALLELISM_SPECS_CLS.TRASNFORMER_LAYER_CLS)
            for _, p in mod.named_parameters(remove_duplicate=remove_duplicate)
        }
        parameter_outside_of_transformer_layers_names = {
            name
            for name, param in model.named_parameters(remove_duplicate=remove_duplicate)
            if param not in parameters_inside_transformer_layers
        }
        return parameter_names | parameter_outside_of_transformer_layers_names

    @abstractclassmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional[torch.device] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
        should_parallelize_layer_predicate_func: Optional[Callable[[torch.nn.Module], bool]] = None,
        **parallel_layer_specific_kwargs,
    ) -> "PreTrainedModel":
        """
        Parallelizes the model by transforming regular layer into their parallel counterparts.
        Each concrete class must implement it.

        Args:
            model (`PreTrainedModel`):
                The model to parallelize.
            device (`Optional[torch.device]`, defaults to `None`):
                The device where the new parallel layers should be put.
            parallelize_embeddings (`bool`, defaults to `True`):
                Whether or not the embeddings should be parallelized.
                This can be disabled in the case when the TP size does not divide the vocabulary size.
            sequence_parallel_enabled (`bool`, defaults to `False`):
                Whether or not sequence parallelism is enabled.
            should_parallelize_layer_predicate_func (Optional[Callable[[torch.nn.Module], bool]], defaults to `None`):
                A function that takes a layer as input and returns a boolean specifying if the input layer should be
                parallelized. This is useful to skip unnecessary parallelization, for pipeline parallelism for instance.
            **parallel_layer_specific_kwargs (`Dict[str, Any]`):
                Keyword arguments specific to some parallel layers, they will be ignored by the other parallel layers.
        Returns:
            `PreTrainedModel`: The parallelized model.
        """
        pass

    @classmethod
    @requires_neuronx_distributed
    def _maybe_load_weights_to_parallel_linears(cls, model: "PreTrainedModel"):
        from neuronx_distributed.parallel_layers.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        weight_map = getattr(model, "_weight_map", {})
        name_to_module = dict(model.named_modules())

        gqa_output_projections = {}
        for fully_qualified_name, layer in name_to_module.items():
            if isinstance(layer, OptimumGQAQKVColumnParallelLinear):
                parent_name = fully_qualified_name.rsplit(".", maxsplit=1)[0]
                output_projection_name = f"{parent_name}.{layer.output_proj_name}"
                gqa_output_projections[output_projection_name] = (
                    layer.num_attention_heads,
                    layer.num_key_value_heads,
                    layer.kv_size_multiplier,
                )

        for fully_qualified_name, layer in name_to_module.items():
            if isinstance(layer, (RowParallelLinear, ColumnParallelLinear)):
                linear_weight_info, linear_bias_weight_info = get_linear_weight_info(
                    weight_map, fully_qualified_name, fail_if_not_found=False
                )
                if linear_weight_info is not None:
                    if fully_qualified_name in gqa_output_projections:
                        num_attention_heads, num_key_value_heads, kv_size_multiplier = gqa_output_projections[
                            fully_qualified_name
                        ]
                        maybe_load_weights_to_output_projection_when_using_gqa_qkv_column_parallel_linear(
                            layer,
                            num_attention_heads,
                            num_key_value_heads,
                            kv_size_multiplier,
                            linear_layer_weight_info=linear_weight_info,
                            linear_layer_bias_weight_info=linear_bias_weight_info,
                        )
                    else:
                        maybe_load_linear_weight_to_parallel_linear(
                            layer,
                            linear_layer_weight_info=linear_weight_info,
                            linear_layer_bias_weight_info=linear_bias_weight_info,
                        )
            elif isinstance(layer, OptimumGQAQKVColumnParallelLinear):
                maybe_load_weights_to_gqa_qkv_column_parallel_linear(model, layer)

    @classmethod
    @requires_neuronx_distributed
    def _initialize_or_load_weights(
        cls,
        model: "PreTrainedModel",
        names_of_the_parameters_to_consider: Set[str],
        device: Optional[torch.device] = None,
    ):
        from neuronx_distributed import parallel_layers
        from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
        from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank
        from neuronx_distributed.parallel_layers.utils import copy_tensor_model_parallel_attributes

        weight_map = getattr(model, "_weight_map", {})
        with torch.no_grad():
            tied_weights = {}
            new_parameters = set()
            modules_to_initialize = defaultdict(list)
            for name, parameter in model.named_parameters(remove_duplicate=False):
                split = name.rsplit(".", maxsplit=1)
                module = model.get_submodule(split[0])
                attribute_name = split[1]

                # Skipping the parameters that will not end-up in this pipeline rank.
                if name not in names_of_the_parameters_to_consider:
                    continue

                try:
                    weight_info = WeightInformation(weight_map[name], name, weight_map=weight_map, device=device)
                except KeyError:
                    weight_info = None

                if parameter in new_parameters:
                    # It can be the case if a module is shared in the model.
                    # For example in T5, the embedding layer is shared so after loading the parameter the first time,
                    # it is not needed to do it again, and doing it can cause bugs.
                    continue
                elif parameter in tied_weights:
                    # It can be the case when weights are tied. For example between the embeddings and the LM head.
                    new_parameter = tied_weights[parameter]
                elif weight_info is not None:
                    if getattr(parameter, "tensor_model_parallel", False):
                        if parameter.device == torch.device(
                            "meta"
                        ) or not was_already_initialized_during_parallelization(parameter):
                            # This must either be a torch.nn.Embedding or a torch.nn.Linear that was not handled during
                            # parallelization since those are the only classes that we initialize on the `meta` device.
                            num_dims = parameter.dim()
                            partition_dim = getattr(parameter, "partition_dim")
                            tp_rank = get_tensor_model_parallel_rank()
                            size_per_rank = parameter.size(partition_dim)
                            slices = [
                                (
                                    None
                                    if idx != partition_dim
                                    else (size_per_rank * tp_rank, size_per_rank * (tp_rank + 1))
                                )
                                for idx in range(num_dims)
                            ]
                        else:
                            # The parameter is not on the `meta` device, it has been loaded from a checkpoint during
                            # parallelization, we can skip.
                            tied_weights[parameter] = parameter
                            new_parameters.add(parameter)
                            continue
                    else:
                        slices = None
                    weight_data = load_tensor_for_weight(weight_info, tensor_slices=slices).to(parameter.dtype)
                    if device is not None:
                        weight_data = weight_data.to(device)
                    new_parameter = torch.nn.Parameter(weight_data)
                    copy_tensor_model_parallel_attributes(new_parameter, parameter)
                elif parameter.device != torch.device("meta") and (
                    was_already_initialized_during_parallelization(parameter)
                    or not parameter_can_be_initialized(model, module, attribute_name)
                ):
                    tied_weights[parameter] = parameter
                    new_parameters.add(parameter)
                    continue
                else:
                    # This means that there is no information about where to find the weights for this parameter.
                    # We first create the module on CPU, initialize it and then move it on device if needed.
                    device = torch.device("cpu")
                    new_parameter = torch.nn.Parameter(torch.empty_like(parameter, device=device))
                    copy_tensor_model_parallel_attributes(new_parameter, parameter)
                    modules_to_initialize[module].append(attribute_name)

                setattr(
                    module,
                    attribute_name,
                    new_parameter,
                )
                tied_weights[parameter] = new_parameter
                new_parameters.add(new_parameter)
                gc.collect()

            for mod, parameter_names in modules_to_initialize.items():
                if isinstance(mod, torch.nn.Embedding):
                    # This module has not pre-trained weights, it must be fine-tuned, we initialize it with the
                    # `reset_parameters()` method since there is only one parameter in torch.nn.Embedding.
                    left_uninitialized = try_to_hf_initialize(model, mod, parameter_names)
                    if left_uninitialized:
                        mod.reset_parameters()
                elif isinstance(mod, torch.nn.Linear):
                    # This module has not pre-trained weights, it must be fine-tuned, we initialize it with the
                    # `reset_parameters()` method but we need to be careful because one of the parameters might not
                    # need initialization.
                    left_uninitialized = try_to_hf_initialize(model, mod, parameter_names)
                    if left_uninitialized:
                        initialize_torch_nn_module(mod, left_uninitialized)
                elif isinstance(mod, parallel_layers.layers.BaseParallelLinear):
                    # First, we try to initialize the layer similarly as it would be done with the model.
                    # To do that we initialize a `torch.nn.Linear` with the full shape, and then scatter the weights.
                    input_is_parallel = gather_output = False
                    if isinstance(mod, parallel_layers.layers.RowParallelLinear):
                        axis = "row"
                        input_is_parallel = mod.input_is_parallel
                    elif isinstance(mod, parallel_layers.layers.ColumnParallelLinear):
                        axis = "column"
                        gather_output = mod.gather_output
                    elif isinstance(mod, GQAQKVColumnParallelLinear):
                        axis = "qga_qkv_column"
                        gather_output = mod.gather_output
                    else:
                        raise RuntimeError(
                            f"This kind of parallel linear is not supported yet: {mod.__class__.__name__}"
                        )

                    if axis in ["row", "column"]:
                        fake_linear_mod = torch.nn.Linear(mod.input_size, mod.output_size)
                        left_uninitialized = try_to_hf_initialize(model, fake_linear_mod, parameter_names)
                        if left_uninitialized:
                            initialize_parallel_linear(mod, left_uninitialized)
                        else:
                            fake_parallel_linear_mod = linear_to_parallel_linear(
                                fake_linear_mod,
                                axis,
                                input_is_parallel=input_is_parallel,
                                gather_output=gather_output,
                                sequence_parallel_enabled=mod.sequence_parallel_enabled,
                            )
                            mod.weight.copy_(fake_parallel_linear_mod.weight.data)
                            if mod.bias is not None:
                                mod.bias.copy_(fake_parallel_linear_mod.bias.data)
                            del fake_parallel_linear_mod
                        del fake_linear_mod
                    else:

                        def initialize(mod: GQAQKVColumnParallelLinear, proj_name: str, output_size: int):
                            fake_linear_mod = torch.nn.Linear(mod.input_size, output_size)
                            parameter_names_to_consider = [
                                name for name in parameter_names if name.endswith(f"_{proj_name}")
                            ]
                            mapping = {
                                f"weight_{proj_name}": "weight",
                                f"bias_{proj_name}": "bias",
                            }
                            left_uninitialized = try_to_hf_initialize(
                                model, fake_linear_mod, parameter_names_to_consider, parameter_names_mapping=mapping
                            )
                            if left_uninitialized:
                                initialize_parallel_linear(mod, left_uninitialized)
                            else:
                                # TODO: change kv heads.
                                maybe_load_linear_weight_to_gqa_qkv_column_parallel_linear(
                                    mod, proj_name, f"weight_{proj_name}", linear_layer=fake_linear_mod
                                )
                            del fake_linear_mod

                        initialize(mod, "q", mod.output_sizes[0])
                        initialize(mod, "k", mod.output_sizes[1])
                        initialize(mod, "v", mod.output_sizes[1])
                else:
                    # TODO: maybe we should not allow that. Since the module should already be initialized because it
                    # is ignored by lazy loading.
                    left_uninitialized = try_to_hf_initialize(model, mod, parameter_names)
                    if left_uninitialized and hasattr(mod, "reset_parameters"):
                        initialize_torch_nn_module(mod, parameter_names)

                gc.collect()

    @classmethod
    @requires_neuronx_distributed
    def parallelize(
        cls,
        model: Union["PreTrainedModel", NeuronPeftModel],
        device: Optional[torch.device] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
        kv_size_multiplier: Optional[int] = None,
        pipeline_parallel_input_names: Optional[Union[Tuple[str, ...], Dict[str, Tuple[str, ...]]]] = None,
        pipeline_parallel_num_microbatches: int = 1,
        pipeline_parallel_use_zero1_optimizer: bool = False,
        pipeline_parallel_gradient_checkpointing_enabled: bool = False,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        num_local_ranks_per_step: int = 8,
    ) -> "PreTrainedModel":
        """
        Parallelizes the model by transforming regular layer into their parallel counterparts using
        `cls._parallelize()`.

        It also makes sure that each parameter has loaded its weights or has been initialized if there is no pre-trained
        weights associated to it.

        Args:
            model (`Union[PreTrainedModel, NeuronPeftModel]`):
                The model to parallelize.
            device (`Optional[torch.device]`, defaults to `None`):
                The device where the new parallel layers should be put.
            parallelize_embeddings (`bool`, defaults to `True`):
                Whether or not the embeddings should be parallelized.
                This can be disabled in the case when the TP size does not divide the vocabulary size.
            sequence_parallel_enabled (`bool`, defaults to `False`):
                Whether or not sequence parallelism is enabled.
            kv_size_multiplier (`Optional[int], defaults to `None`):
                The number of times to replicate the KV heads when the TP size is bigger than the number of KV heads.
                If left unspecified, the smallest multiplier that makes the number of KV heads divisible by the TP size
                will be used.
            pipeline_parallel_num_microbatches (`int`, defaults to 1):
                The number of microbatches used for pipeline execution.
            pipeline_parallel_use_zero1_optimizer (`bool`, defaults to `False`):
                When zero-1 optimizer is used, set this to True, so the PP model will understand that zero-1 optimizer
                will handle data parallel gradient averaging.
            pipeline_parallel_gradient_checkpointing_enabled (`bool`, defaults to `False`):
                Whether or not gradient checkpointing should be enabled when doing pipeline parallelism.
            checkpoint_dir (`Optional[Union[str, Path]]`):
                Path to a sharded checkpoint. If specified, the checkpoint weights will be loaded to the parallelized
                model.
            num_local_ranks_per_step (`int`, defaults to `8`):
                Corresponds to the number of local ranks that can initialize and load the model weights at the same
                time. If the value is inferior to 0, the maximum number of ranks will be used.

        Returns:
            `PreTrainedModel`: The parallelized model.
        """
        import torch_xla.core.xla_model as xm

        orig_model, peft_prefix = get_base_model_and_peft_prefix(model)
        model_class = orig_model.__class__

        if peft_prefix:
            # We update the weight_map to contain both the original parameter names, and the ones in the PeftModel.
            # The reason we keep both is because depending on the context during parallelization one or the other name
            # will be used. Since the names with prefix should not overwrite anything, it is safe to have both.
            if hasattr(orig_model, "_weight_map"):
                weight_map = orig_model._weight_map
                weight_map["peft_prefix"] = peft_prefix
                peft_model_weight_map = {f"{peft_prefix}.{name}": filename for name, filename in weight_map.items()}
                for name, _ in model.named_parameters():
                    name_without_base_layer = name.replace(".base_layer", "")
                    if name not in peft_model_weight_map and name_without_base_layer in peft_model_weight_map:
                        peft_model_weight_map[name] = peft_model_weight_map.pop(name_without_base_layer)
                weight_map.update(**peft_model_weight_map)

        if sequence_parallel_enabled and not cls.supports_sequence_parallelism():
            raise NotImplementedError(f"Sequence parallelism is not supported for {model_class}.")

        from neuronx_distributed.parallel_layers.parallel_state import (
            get_pipeline_model_parallel_size,
            get_tensor_model_parallel_size,
        )
        from neuronx_distributed.parallel_layers.random import _MODEL_PARALLEL_RNG_TRACKER_NAME, get_xla_rng_tracker
        from neuronx_distributed.parallel_layers.utils import get_local_world_size
        from neuronx_distributed.pipeline import NxDPPModel

        tp_size = get_tensor_model_parallel_size()
        pp_size = get_pipeline_model_parallel_size()

        sequence_parallel_enabled = sequence_parallel_enabled and tp_size > 1

        # Parallelizing the model.
        # This needs to be done prior to preparing the model for sequence parallelism because modules can be overriden.
        name_to_parameter = dict(model.named_parameters(remove_duplicate=False))
        parameter_to_name = {p: n for n, p in name_to_parameter.items()}

        names_of_the_parameters_to_consider = cls._get_parameter_names_for_current_pipeline(
            model, remove_duplicate=True
        )

        if peft_prefix:
            names_of_the_parameters_to_consider = {
                f"{peft_prefix}.{name}" for name in names_of_the_parameters_to_consider
            }

        # We delay weight loading when the model was instantiated from pretrained lazily.
        # We do not skip for cases such as:
        #   - Loaded a model `from_config`: in this case we simply initialize later in `_initialize_or_load_weights`.
        #   - Loaded a model `from_pretrained` but not lazily.
        skip_linear_weight_load = hasattr(model, "_weight_map")

        requires_grad_information = {n: p.requires_grad for n, p in model.named_parameters()}

        def should_parallelize_layer_predicate_func(layer):
            if pp_size == 1:
                return True
            for p in layer.parameters():
                if p not in parameter_to_name:
                    return True
            names = {parameter_to_name[p] for p in layer.parameters()}
            return names < names_of_the_parameters_to_consider

        if tp_size > 1:
            # TODO: remove that once it is solved on the `neuronx_distributed` side.
            try:
                get_xla_rng_tracker().add(_MODEL_PARALLEL_RNG_TRACKER_NAME, 42)
            except Exception:
                # It means that `_MODEL_PARALLEL_RNG_TRACKER_NAME` was already added to the rng tracker, we can ignore.
                pass

            tied_parameters = get_tied_parameters_dict(model)

            cls._parallelize(
                orig_model,
                device=device,
                parallelize_embeddings=parallelize_embeddings,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                skip_linear_weight_load=skip_linear_weight_load,
                kv_size_multiplier=kv_size_multiplier,
            )

            for param_name, root_tied_param_name in tied_parameters.items():
                parent_mod, attr_name = get_parent_module_and_param_name_from_fully_qualified_name(model, param_name)
                param = getattr(parent_mod, attr_name)
                root_parent_mod, root_attr_name = get_parent_module_and_param_name_from_fully_qualified_name(
                    model, root_tied_param_name
                )
                root_tied_param = getattr(root_parent_mod, root_attr_name)
                if getattr(param, "tensor_model_parallel", False) and not getattr(
                    root_tied_param, "tensor_model_parallel", False
                ):
                    # In this case it means that `param` was parallelized but not `root_tied_param`.
                    # It will be overwritten by root_tied_param when tiying the weights if we do not do anything.
                    # We tie `root_tied_param` to `param`.
                    # What will happen is as follows:
                    #   - If weight_map contains a checkpoint for `param_name`, it has already been initialized or will
                    #   be initialized with `cls._maybe_load_weights_to_parallel_linear`.
                    #   - Otherwise, it has not been initialized and `cls._initialize_or_load_weights` will take care
                    #   of it.
                    root_parent_base_mod, _ = get_base_model_and_peft_prefix(root_parent_mod)
                    setattr(root_parent_base_mod, root_attr_name, param)

            if is_main_worker():
                logger.info("Tensor parallelism done.")

            # We need to refresh the names because they might have changed after `_parallelize`.
            # For instance if we changed regular linears to GQAQKVColumnParallelLinear.
            names_of_the_parameters_to_consider = cls._get_parameter_names_for_current_pipeline(
                model, remove_duplicate=True
            )

        # We need to retrieve this mapping here because PP works with `torch.fx` so we will not end-up with the same
        # names after tracing.
        gqa_qkv_metadata = {
            "original_names_to_gqa_qkv_names": {},
            "output_projections_names": set(),
            "num_attention_heads": None,
            "num_key_value_heads": None,
            "kv_size_multiplier": None,
            "fuse_qkv": None,
            "q_output_size_per_partition": None,
            "kv_output_size_per_partition": None,
        }
        for mod in model.modules():
            if isinstance(mod, OptimumGQAQKVColumnParallelLinear):
                num_attention_heads = mod.num_attention_heads
                num_key_value_heads = mod.num_key_value_heads
                kv_size_multiplier = mod.kv_size_multiplier
                gqa_qkv_metadata = {
                    "original_names_to_gqa_qkv_names": get_parameter_names_mapping_after_gqa_qkv_replacement(model),
                    "output_projections_names": get_output_projection_qualified_names_after_qga_qkv_replacement(model),
                    "num_attention_heads": num_attention_heads,
                    "num_key_value_heads": num_key_value_heads,
                    "kv_size_multiplier": kv_size_multiplier,
                    "fuse_qkv": mod.fuse_qkv,
                    "q_output_size_per_partition": mod.q_output_size_per_partition,
                    "kv_output_size_per_partition": mod.kv_output_size_per_partition,
                }
                break

        # Preparing the model for sequence parallelism:
        sp_specs_cls = cls.SEQUENCE_PARALLELSIM_SPECS_CLS

        if sequence_parallel_enabled:
            # 1. Transforming the LayerNorms.
            layer_norm_qualified_name_patterns = (
                sp_specs_cls.SEQUENCE_PARALLEL_LAYERNORM_PATTERNS
                if sp_specs_cls.SEQUENCE_PARALLEL_LAYERNORM_PATTERNS is not None
                else []
            )
            sequence_collective_op_infos = sp_specs_cls.SEQUENCE_COLLECTIVE_OPS_INFOS
            if peft_prefix and sequence_collective_op_infos is not None:
                layer_norm_qualified_name_patterns = [
                    f"{peft_prefix}.{pattern}" for pattern in layer_norm_qualified_name_patterns
                ]
                for idx, sp_collective_info in enumerate(sequence_collective_op_infos):
                    if isinstance(sp_collective_info.layer, str):
                        sequence_collective_op_infos[idx] = replace(
                            sp_collective_info, layer=f"{peft_prefix}.{sp_collective_info.layer}"
                        )

            layer_norm_sequence_parallelizer = LayerNormSequenceParallelizer(
                sequence_parallel_enabled, layer_norm_qualified_name_patterns
            )
            layer_norm_sequence_parallelizer.sequence_parallelize(model, sp_specs_cls.LAYERNORM_TYPE)

            # 2. Taking care of scattering / gathering on the sequence axis in the model via the IOSequenceParallelizer.
            io_sequence_parallelizer = IOSequenceParallelizer(
                sequence_parallel_enabled,
                sequence_collective_op_infos=sequence_collective_op_infos,
            )
            io_sequence_parallelizer.sequence_parallelize(model)

            # 3. Applying model specific patching for sequence parallelism.
            sp_specs_cls.patch_for_sequence_parallelism(model, sequence_parallel_enabled)

        if is_main_worker():
            logger.info("Loading and initializing the weights, this might take a while on large models.")

        local_rank = xm.get_local_ordinal()
        if num_local_ranks_per_step <= 0:
            num_local_ranks_per_step = get_local_world_size()
        for worker in range(math.ceil(get_local_world_size() / num_local_ranks_per_step)):
            if local_rank // num_local_ranks_per_step == worker:
                if skip_linear_weight_load:
                    # Load the weights to the parallel linears if the loading was skipped during parallelization.
                    cls._maybe_load_weights_to_parallel_linears(model)

                if skip_linear_weight_load or any(p.device == torch.device("meta") for p in model.parameters()):
                    # Initialize or load the weights for the parallelized model if it was lazily loaded.
                    cls._initialize_or_load_weights(model, names_of_the_parameters_to_consider, device=device)
            gc.collect()

        # It is important to do that here because initialization can untie weights.
        model.tie_weights()

        # Because we initialize new parameters, we need to make sure that only the ones that required grads before
        # parallelization require grad after parallelization.
        for name, parameter in model.named_parameters():
            gqa_qkv_names_to_original_names = {
                v: k for k, v in gqa_qkv_metadata["original_names_to_gqa_qkv_names"].items()
            }
            if name in requires_grad_information:
                parameter.requires_grad = requires_grad_information[name]
            elif gqa_qkv_names_to_original_names.get(name, None) in requires_grad_information:
                gqa_qkv_name = gqa_qkv_names_to_original_names[name]
                parameter.requires_grad = requires_grad_information[gqa_qkv_name]
            else:
                raise ValueError(
                    f"Could not find information for the parameter {name} to set its `requires_grad` attribute."
                )

        if is_main_worker():
            logger.info("Load and initialization of the weights done.")

        if pp_size > 1:
            if isinstance(model, NeuronPeftModel):
                raise NotImplementedError("PEFT is not supported with model parallelism for now.")

            if not cls.supports_pipeline_parallelism():
                raise NotImplementedError("{cls} does not support pipeline parallelism.")

            model.config.use_cache = False
            model.config.return_dict = False
            model.config.output_attentions = False
            model.config.output_hidden_states = False

            with Patcher(cls.PIPELINE_PARALLELISM_SPECS_CLS.get_patching_specs()):
                if pipeline_parallel_input_names is None:
                    pipeline_parallel_input_names = cls.PIPELINE_PARALLELISM_SPECS_CLS.DEFAULT_INPUT_NAMES

                if isinstance(pipeline_parallel_input_names, dict):
                    if model_class.__name__ in pipeline_parallel_input_names:
                        pipeline_parallel_input_names = pipeline_parallel_input_names[model_class.__name__]
                    elif "default" in pipeline_parallel_input_names:
                        pipeline_parallel_input_names = pipeline_parallel_input_names["default"]
                    else:
                        raise ValueError(
                            "Cannot guess the names of the input for the model, which is required for pipeline "
                            "parallelism."
                        )

                model = NxDPPModel(
                    model,
                    transformer_layer_cls=cls.PIPELINE_PARALLELISM_SPECS_CLS.TRASNFORMER_LAYER_CLS,
                    num_microbatches=pipeline_parallel_num_microbatches,
                    output_loss_value_spec=cls.PIPELINE_PARALLELISM_SPECS_CLS.OUTPUT_LOSS_SPECS,
                    input_names=pipeline_parallel_input_names,
                    pipeline_cuts=cls.PIPELINE_PARALLELISM_SPECS_CLS.create_pipeline_cuts(model, pp_size),
                    leaf_module_cls=cls.PIPELINE_PARALLELISM_SPECS_CLS.leaf_module_cls(),
                    use_zero1_optimizer=pipeline_parallel_use_zero1_optimizer,
                    tracer_cls=OptimumNeuronFXTracer,
                )

            if is_main_worker():
                logger.info("Pipeline parallelism done.")

        # TODO: can we optimize by skipping initialization and weight loading when `checkpoint_dir` is not None.
        if not is_precompilation() and checkpoint_dir is not None:
            cls.load_model_sharded_checkpoint(model, checkpoint_dir)

        model._gqa_qkv_metadata = gqa_qkv_metadata

        return model

    @classmethod
    def deparallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        raise NotImplementedError

    @classmethod
    @requires_neuronx_distributed
    def was_parallelized(cls, model: "PreTrainedModel") -> bool:
        import neuronx_distributed
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_pipeline_model_parallel_size,
            get_tensor_model_parallel_size,
        )
        from neuronx_distributed.pipeline import NxDPPModel

        needs_parallelization_for_pp = get_pipeline_model_parallel_size() > 1 and not isinstance(model, NxDPPModel)
        parallel_layer_classes = (
            neuronx_distributed.parallel_layers.ParallelEmbedding,
            neuronx_distributed.parallel_layers.ColumnParallelLinear,
            neuronx_distributed.parallel_layers.RowParallelLinear,
        )
        layers_are_parallel = any(isinstance(mod, parallel_layer_classes) for mod in model.modules())
        needs_parallelization_for_tp = get_tensor_model_parallel_size() > 1 and not layers_are_parallel
        return (not needs_parallelization_for_pp) and (not needs_parallelization_for_tp)

    @classmethod
    def _check_model_was_parallelized(cls, model: "PreTrainedModel"):
        if not cls.was_parallelized(model):
            raise ValueError("The model needs to be parallelized first.")

    @classmethod
    @requires_torch_xla
    def optimizer_cpu_params_to_xla_params(
        cls,
        optimizer: "torch.optim.Optimizer",
        orig_param_to_parallel_param_on_xla: Mapping[int, "torch.nn.Parameter"],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        import torch_xla.core.xla_model as xm

        parameters_on_xla = []
        need_to_create_new_optimizer = False
        if hasattr(optimizer, "_args_to_recreate"):
            args, _ = optimizer._args_to_recreate

            # parameter_groups can either be an iterable of dictionaries (groups), or of parameters, in which case
            # there is only one group.
            parameter_groups = args[0]
            parameter_groups = list(parameter_groups)
            # parameter_groups cannot be empty
            if isinstance(parameter_groups[0], dict):
                for group in parameter_groups:
                    new_group = {k: v for k, v in group.items() if k != "params"}
                    params_on_xla = []
                    for p in group["params"]:
                        if p.device == xm.xla_device():
                            params_on_xla.append(p)
                        elif id(p) not in orig_param_to_parallel_param_on_xla:
                            # This can be the case with pipeline parallelism.
                            continue
                        else:
                            params_on_xla.append(orig_param_to_parallel_param_on_xla[id(p)])
                    new_group["params"] = params_on_xla
                    parameters_on_xla.append(new_group)
            else:
                new_param = {}
                params_on_xla = []
                for param in parameter_groups:
                    if param.device == xm.xla_device():
                        params_on_xla.append(param)
                    elif id(param) not in orig_param_to_parallel_param_on_xla:
                        # This can be the case with pipeline parallelism.
                        continue
                    else:
                        params_on_xla.append(orig_param_to_parallel_param_on_xla[id(param)])
                new_param["params"] = params_on_xla
                parameters_on_xla.append(new_param)
        else:
            for param_group in optimizer.param_groups:
                new_params = []
                params = param_group["params"]
                for idx in range(len(params)):
                    if params[idx].device == xm.xla_device():
                        param_on_xla = params[idx]
                    elif id(params[idx]) not in orig_param_to_parallel_param_on_xla:
                        need_to_create_new_optimizer = True
                        continue
                    else:
                        param_on_xla = orig_param_to_parallel_param_on_xla[id(params[idx])]
                    if params[idx] is not param_on_xla:
                        need_to_create_new_optimizer = True
                    new_params.append(param_on_xla)
                new_group = {k: v for k, v in param_group.items() if k != "params"}
                new_group["params"] = new_params
                parameters_on_xla.append(new_group)
        return parameters_on_xla, need_to_create_new_optimizer

    @classmethod
    def optimizer_for_mp(
        cls,
        optimizer: "torch.optim.Optimizer",
        orig_param_to_parallel_param_on_xla: Mapping[int, "torch.nn.Parameter"],
    ) -> "torch.optim.Optimizer":
        """
        Creates an optimizer ready for a parallelized model from an existing optimizer.

        There are two cases:
            1. The optimizer has been created via a lazy constructor from
            [`optimum.neuron.distributed.utils.make_optimizer_constructor_lazy`], it which case the exactly intended optimizer is
            created for tensor parallelism.
            2. The optimizer was created with a regular constructor. In this case the optimizer for tensor parallelism
            is created as close as possible to what was intended but that is not guaranteed.

        Args:
            optimizer (`torch.optim.Optimizer`):
                The original optimizer.
            orig_param_to_parallel_param_on_xla (`Mapping[int, torch.nn.Parameter]`):
                A mapping (e.g. dict-like) that maps the id of a parameter in `optimizer` to the id of its
                parallelized counterpart on an XLA device.

        Returns:
            `torch.optim.Optimizer`: The tensor parallelism ready optimizer.
        """
        parallel_parameters, need_to_create_new_optimizer = cls.optimizer_cpu_params_to_xla_params(
            optimizer, orig_param_to_parallel_param_on_xla
        )
        if hasattr(optimizer, "_args_to_recreate"):
            args, kwargs = optimizer._args_to_recreate
            optimizer_for_mp = optimizer.__class__(parallel_parameters, *args[1:], **kwargs)
            del optimizer
        elif need_to_create_new_optimizer:
            optimizer_for_mp = optimizer.__class__(parallel_parameters)
            del optimizer
        else:
            optimizer_for_mp = optimizer
        return optimizer_for_mp

    @classmethod
    @requires_neuronx_distributed
    def save_model_sharded_checkpoint(
        cls,
        model: Union["PreTrainedModel", "NxDPPModel"],
        output_dir: Union[str, Path],
        optimizer: Optional["torch.optim.Optimizer"] = None,
        use_xser: bool = True,
        async_save: bool = False,
        num_local_ranks_per_step: int = 8,
    ):
        import neuronx_distributed
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_data_parallel_rank,
            get_pipeline_model_parallel_rank,
            get_tensor_model_parallel_rank,
        )
        from neuronx_distributed.parallel_layers.utils import get_local_world_size

        cls._check_model_was_parallelized(model)

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if num_local_ranks_per_step <= 0:
            num_local_ranks_per_step = get_local_world_size()

        metadata = {}
        metadata["sharded_metadata"] = {
            k: asdict(v) for k, v in get_parameters_tp_metadata(dict(model.named_parameters())).items()
        }
        metadata["gqa_qkv_metadata"] = model._gqa_qkv_metadata

        neuronx_distributed.trainer.save_checkpoint(
            output_dir.as_posix(),
            tag=MODEL_PARALLEL_SHARDS_DIR_NAME,
            model=model,
            optimizer=optimizer,
            use_xser=use_xser,
            async_save=async_save,
            num_workers=num_local_ranks_per_step,
        )

        if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
            pp_rank = get_pipeline_model_parallel_rank()
            metadata_path = output_dir / MODEL_PARALLEL_SHARDS_DIR_NAME / f"mp_metadata_pp_rank_{pp_rank}.pt"
            # Checking that the parent directory exists, it should exist, but let's make sure since g_iostate.end() is
            # called at the end of `neuronx_distributed.trainer.save_checkpoint` and it can remove checkpoint
            # directories if the max limit has been reached.
            if metadata_path.parent.is_dir():
                torch.save(metadata, metadata_path)

    @classmethod
    @requires_neuronx_distributed
    def load_sharded_checkpoint(
        cls,
        load_dir: Union[str, Path],
        model: Optional["PreTrainedModel"] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        import neuronx_distributed

        if model is None and optimizer is None:
            raise ValueError("At least a model or an optimizer must be provided")

        if model is not None:
            cls._check_model_was_parallelized(model)

        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)

        if not (load_dir / MODEL_PARALLEL_SHARDS_DIR_NAME).is_dir():
            raise FileNotFoundError(f"Could not find a sharded checkpoint directory under {load_dir.as_posix()}.")

        neuronx_distributed.trainer.load_checkpoint(
            load_dir.as_posix(),
            tag=MODEL_PARALLEL_SHARDS_DIR_NAME,
            model=model,
            optimizer=optimizer,
        )

    @classmethod
    def load_model_sharded_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        return cls.load_sharded_checkpoint(load_dir, model=model)

    @classmethod
    def load_optimizer_sharded_checkpoint(cls, optimizer: "torch.optim.Optimizer", load_dir: Union[str, Path]):
        return cls.load_sharded_checkpoint(load_dir, optimizer=optimizer)
