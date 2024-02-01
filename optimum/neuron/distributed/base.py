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
import shutil
from abc import ABC, abstractclassmethod
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Set, Tuple, Type, Union

import torch
from transformers import PreTrainedModel
from transformers.utils import WEIGHTS_NAME

from ...utils import logging
from ..utils import is_neuronx_distributed_available, is_torch_xla_available
from ..utils.patching import Patcher
from ..utils.require_utils import requires_neuronx_distributed, requires_torch_xla
from .parallel_layers import (
    IOSequenceParallelizer,
    LayerNormSequenceParallelizer,
    LayerNormType,
    SequenceCollectiveOpInfo,
)
from .utils import (
    TENSOR_PARALLEL_SHARDS_DIR_NAME,
    ParameterMetadata,
    WeightInformation,
    initialize_parallel_linear,
    initialize_torch_nn_module,
    linear_to_parallel_linear,
    load_tensor_for_weight,
    named_parameters,
    parameter_can_be_initialized,
    try_to_hf_initialize,
    was_already_initialized_during_parallelization,
)


if TYPE_CHECKING:
    if is_neuronx_distributed_available():
        from neuronx_distributed.pipeline import NxDPPModel

logger = logging.get_logger()


class SavedModelInTemporaryDirectory:
    def __init__(self, model: "PreTrainedModel"):
        self.tmpdir = TemporaryDirectory()
        self.model = model

    def __enter__(self):
        self.model.save_pretrained(self.tmpdir.name)
        return self.tmpdir.name

    def __exit__(self, *exc):
        self.tmpdir.cleanup()


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
    TRASNFORMER_LAYER_CLS: Type["torch.nn.Module"]
    DEFAULT_INPUT_NAMES: Tuple[str, ...]
    LEAF_MODULE_CLASSES_NAMES: Optional[List[Union[str, Type["torch.nn.Module"]]]] = None
    OUTPUT_LOSS_SPECS: Tuple[bool, ...] = (True, False)

    @classmethod
    @requires_torch_xla
    def create_pipeline_cuts(cls, model: PreTrainedModel, pipeline_parallel_size: int) -> List[str]:
        """
        Creates the pipeline cuts, e.g. the name of the layers at each the cuts happen for pipeline parallelism.
        """
        import torch_xla.core.xla_model as xm

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

        if xm.get_local_ordinal() == 0:
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
        cls, model: "torch.nn.Module", remove_duplicate: bool = True
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
        all_parameter_names = {n for n, _ in named_parameters(model, remove_duplicate=remove_duplicate)}
        if pp_size == 1:
            return all_parameter_names

        if not cls.supports_pipeline_parallelism():
            raise NotImplementedError(f"{cls} does not support pipeline parallelism.")

        cuts = cls.PIPELINE_PARALLELISM_SPECS_CLS.create_pipeline_cuts(model, pp_size)

        start_module_name = cuts[pp_rank - 1] if pp_rank >= 1 else None
        end_module_name = None if pp_rank == pp_size - 1 else cuts[pp_rank]
        parameter2name = {p: n for n, p in named_parameters(model, remove_duplicate=remove_duplicate)}
        parameter_names = set()
        should_add = False
        for name, mod in model.named_modules():
            if not isinstance(mod, cls.PIPELINE_PARALLELISM_SPECS_CLS.TRASNFORMER_LAYER_CLS):
                continue
            # If start_module_name is None, it means we are on the first rank, we should add right from the beginning.
            if start_module_name is None:
                should_add = True
            if should_add:
                for _, param in named_parameters(mod, remove_duplicate=remove_duplicate):
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
            for _, p in named_parameters(mod, remove_duplicate=remove_duplicate)
        }
        parameter_outside_of_transformer_layers_names = {
            name
            for name, param in named_parameters(model, remove_duplicate=remove_duplicate)
            if param not in parameters_inside_transformer_layers
        }
        return parameter_names | parameter_outside_of_transformer_layers_names

    @abstractclassmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
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
        Returns:
            `PreTrainedModel`: The parallelized model.
        """

    @classmethod
    @requires_neuronx_distributed
    def parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
        pipeline_parallel_input_names: Optional[Union[Tuple[str, ...], List[str]]] = None,
        pipeline_parallel_num_microbatches: int = 1,
        pipeline_parallel_use_zero1_optimizer: bool = False,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ) -> "PreTrainedModel":
        """
        Parallelizes the model by transforming regular layer into their parallel counterparts using
        `cls._parallelize()`.

        It also makes sure that each parameter has loaded its weights or has been initialized if there is no pre-trained
        weights associated to it.

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
            pipeline_parallel_num_microbatches (`int`, defaults to 1):
                The number of microbatches used for pipeline execution.
            pipeline_parallel_use_zero1_optimizer (`bool`, defaults to `False`):
                When zero-1 optimizer is used, set this to True, so the PP model will understand that zero-1 optimizer
                will handle data parallel gradient averaging.
            checkpoint_dir (`Optional[Union[str, Path]]`):
                Path to a sharded checkpoint. If specified, the checkpoint weights will be loaded to the parallelized
                model.

        Returns:
            `PreTrainedModel`: The parallelized model.
        """
        from neuronx_distributed import parallel_layers

        if sequence_parallel_enabled and not cls.supports_sequence_parallelism():
            raise NotImplementedError(f"Sequence parallelism is not supported for {model.__class__}.")

        from neuronx_distributed.parallel_layers.parallel_state import (
            get_pipeline_model_parallel_size,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_size,
        )
        from neuronx_distributed.pipeline import NxDPPModel

        tp_size = get_tensor_model_parallel_size()

        sequence_parallel_enabled = sequence_parallel_enabled and tp_size > 1

        # Parallelizing the model.
        # This needs to be done prior to preparing the model for sequence parallelism because modules can be overriden.
        if tp_size > 1:
            model = cls._parallelize(
                model,
                device=device,
                parallelize_embeddings=parallelize_embeddings,
                sequence_parallel_enabled=sequence_parallel_enabled,
            )

        # Preparing the model for sequence parallelism:
        sp_specs_cls = cls.SEQUENCE_PARALLELSIM_SPECS_CLS

        if sequence_parallel_enabled:
            # 1. Transforming the LayerNorms.
            layer_norm_qualified_name_patterns = (
                sp_specs_cls.SEQUENCE_PARALLEL_LAYERNORM_PATTERNS
                if sp_specs_cls.SEQUENCE_PARALLEL_LAYERNORM_PATTERNS is not None
                else []
            )
            layer_norm_sequence_parallelizer = LayerNormSequenceParallelizer(
                sequence_parallel_enabled, layer_norm_qualified_name_patterns
            )
            layer_norm_sequence_parallelizer.sequence_parallelize(model, sp_specs_cls.LAYERNORM_TYPE)

            # 2. Taking care of scattering / gathering on the sequence axis in the model via the IOSequenceParallelizer.
            io_sequence_parallelizer = IOSequenceParallelizer(
                sequence_parallel_enabled,
                sequence_collective_op_infos=sp_specs_cls.SEQUENCE_COLLECTIVE_OPS_INFOS,
            )
            io_sequence_parallelizer.sequence_parallelize(model)

            # 3. Applying model specific patching for sequence parallelism.
            sp_specs_cls.patch_for_sequence_parallelism(model, sequence_parallel_enabled)

        # The model was not loaded lazily, it is already ready.
        weight_map = getattr(model, "_weight_map", {})

        names_of_the_parameters_to_consider = cls._get_parameter_names_for_current_pipeline(
            model, remove_duplicate=True
        )

        with torch.no_grad():
            tied_weights = {}
            new_parameters = set()
            modules_to_initialize = defaultdict(list)
            for name, parameter in named_parameters(model, remove_duplicate=False):
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
                        if parameter.device == torch.device("meta"):
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

                    new_parameter = torch.nn.Parameter(
                        load_tensor_for_weight(weight_info, tensor_slices=slices).to(parameter.dtype)
                    )
                elif parameter.device != torch.device("meta") and (
                    was_already_initialized_during_parallelization(parameter)
                    or not parameter_can_be_initialized(model, module, attribute_name)
                ):
                    tied_weights[parameter] = parameter
                    new_parameters.add(parameter)
                    continue
                else:
                    # This means that there is no information about where to find the weights for this parameter.
                    device = torch.device("cpu") if device is None else device
                    new_parameter = torch.nn.Parameter(torch.empty_like(parameter, device=device))
                    modules_to_initialize[module].append(attribute_name)

                setattr(
                    module,
                    attribute_name,
                    new_parameter,
                )
                tied_weights[parameter] = new_parameter
                new_parameters.add(new_parameter)

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
                    else:
                        axis = "column"
                        gather_output = mod.gather_output
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
                        mod.weight.data = fake_parallel_linear_mod.weight.data.clone()
                        if mod.bias is not None:
                            mod.bias.data = fake_parallel_linear_mod.bias.data.clone()
                        del fake_linear_mod
                        del fake_parallel_linear_mod
                else:
                    left_uninitialized = try_to_hf_initialize(model, mod, parameter_names)
                    if left_uninitialized and hasattr(mod, "reset_parameters"):
                        initialize_torch_nn_module(mod, parameter_names)

        pp_size = get_pipeline_model_parallel_size()
        if pp_size > 1:
            if not cls.supports_pipeline_parallelism():
                raise NotImplementedError("{cls} does not support pipeline parallelism.")

            model.config.return_dict = False
            model.config.use_cache = False
            model.config.output_attentions = False
            model.config.output_hidden_states = False

            with Patcher(cls.PIPELINE_PARALLELISM_SPECS_CLS.get_patching_specs()):
                if pipeline_parallel_input_names is None:
                    pipeline_parallel_input_names = cls.PIPELINE_PARALLELISM_SPECS_CLS.DEFAULT_INPUT_NAMES
                model = NxDPPModel(
                    model,
                    transformer_layer_cls=cls.PIPELINE_PARALLELISM_SPECS_CLS.TRASNFORMER_LAYER_CLS,
                    num_microbatches=pipeline_parallel_num_microbatches,
                    output_loss_value_spec=cls.PIPELINE_PARALLELISM_SPECS_CLS.OUTPUT_LOSS_SPECS,
                    input_names=pipeline_parallel_input_names,
                    pipeline_cuts=cls.PIPELINE_PARALLELISM_SPECS_CLS.create_pipeline_cuts(model, pp_size),
                    leaf_module_cls=cls.PIPELINE_PARALLELISM_SPECS_CLS.leaf_module_cls(),
                    use_zero1_optimizer=pipeline_parallel_use_zero1_optimizer,
                )

        if checkpoint_dir is not None:
            cls.load_model_checkpoint(model, checkpoint_dir)

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
    def _get_parameters_tp_metadata(cls, named_parameters: Dict[str, "torch.nn.Parameter"]):
        tp_metadata = {}
        for name, param in named_parameters.items():
            if getattr(param, "tensor_model_parallel", False):
                param_metadata = ParameterMetadata(
                    "sharded",
                    partition_dim=param.partition_dim,
                )
            else:
                param_metadata = ParameterMetadata("tied")
            tp_metadata[name] = param_metadata
        return tp_metadata

    @classmethod
    @requires_neuronx_distributed
    def save_model_checkpoint_as_regular(
        cls,
        model: "PreTrainedModel",
        output_dir: Union[str, Path],
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ):
        import neuronx_distributed
        import torch_xla.core.xla_model as xm
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_data_parallel_rank,
            get_tensor_model_parallel_rank,
        )

        cls._check_model_was_parallelized(model)

        data_parallel_rank = get_data_parallel_rank()
        tensor_parallel_rank = get_tensor_model_parallel_rank()

        if data_parallel_rank != 0:
            return

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if optimizer is not None:
            logger.warning(
                "Saving the optimizer state as a regular file under the tensor parallel setting is not supported yet."
            )

        state_dict = {}
        for name, param in model.named_parameters():
            if getattr(param, "tensor_model_parallel", False):
                if param.partition_dim == 1:
                    tensor = neuronx_distributed.utils.gather_from_tensor_model_parallel_region(param)
                else:
                    # Because the gather works only on last dim. Need to make it work for all dims.
                    tensor = neuronx_distributed.utils.gather_from_tensor_model_parallel_region(
                        param.transpose()
                    ).transpose()
            else:
                tensor = param
            state_dict[name] = tensor

        model_state_dict = {"model": state_dict}
        should_save = tensor_parallel_rank == 0
        xm._maybe_convert_to_cpu(model_state_dict, convert=should_save)
        if should_save:
            output_path = output_dir / WEIGHTS_NAME
            torch.save(model_state_dict["model"], output_path.as_posix())
        xm.rendezvous("saving regular checkpoint")

    @classmethod
    @requires_neuronx_distributed
    def save_model_checkpoint_as_sharded(
        cls,
        model: Union["PreTrainedModel", "NxDPPModel"],
        output_dir: Union[str, Path],
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ):
        import torch_xla.core.xla_model as xm
        from neuronx_distributed import parallel_layers
        from neuronx_distributed.pipeline import NxDPPModel

        cls._check_model_was_parallelized(model)

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if isinstance(model, NxDPPModel):
            model_state_dict = model.local_state_dict()
        else:
            model_state_dict = model.state_dict()

        state_dict = {"model": model_state_dict}
        state_dict["sharded_metadata"] = {
            k: asdict(v) for k, v in cls._get_parameters_tp_metadata(dict(model.named_parameters())).items()
        }

        if optimizer is not None:
            # TODO: have metadata working for the optimizer.
            state_dict["optimizer_state_dict"] = optimizer.state_dict()

        output_path = output_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME

        if xm.get_local_ordinal() == 0:
            if output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
            output_path.mkdir()
        xm.rendezvous("waiting before saving")
        parallel_layers.save(state_dict, output_path.as_posix(), save_xser=True)

    @classmethod
    def save_model_checkpoint(
        cls,
        model: "PreTrainedModel",
        output_dir: Union[str, Path],
        as_regular: bool = False,
        as_sharded: bool = True,
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ):
        if not as_regular and not as_sharded:
            raise ValueError("At least as_regular or as_sharded must be True.")
        if as_regular:
            cls.save_model_checkpoint_as_regular(model, output_dir, optimizer=optimizer)
        if as_sharded:
            cls.save_model_checkpoint_as_sharded(model, output_dir, optimizer=optimizer)

    @classmethod
    @requires_neuronx_distributed
    def load_model_sharded_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        import neuronx_distributed

        cls._check_model_was_parallelized(model)

        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)
        neuronx_distributed.parallel_layers.load(
            load_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME,
            model_or_optimizer=model,
            load_xser=True,
            sharded=True,
        )

    @classmethod
    def load_model_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)

        if (load_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME).is_dir():
            cls.load_model_sharded_checkpoint(model, load_dir)
        else:
            raise FileNotFoundError(f"Could not find a sharded checkpoint directory under {load_dir.as_posix()}.")

    @classmethod
    @requires_neuronx_distributed
    def load_optimizer_sharded_checkpoint(cls, optimizer: "torch.optim.Optimizer", load_dir: Union[str, Path]):
        import neuronx_distributed
        from neuronx_distributed.optimizer import NeuronZero1Optimizer

        is_zero_1_optimizer = optimizer.__class__.__name__ == "NeuronAcceleratedOptimizer" and isinstance(
            optimizer.optimizer, NeuronZero1Optimizer
        )
        is_zero_1_optimizer = is_zero_1_optimizer or isinstance(optimizer, NeuronZero1Optimizer)
        if is_zero_1_optimizer:
            raise NotImplementedError(
                "It is not possible to load a sharded optimizer checkpoint when using ZeRO-1 yet."
            )

        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)

        neuronx_distributed.parallel_layers.load(
            load_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME,
            model_or_optimizer=optimizer,
            model_key="optimizer_state_dict",
            load_xser=True,
            sharded=True,
        )
