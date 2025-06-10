# coding=utf-8
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

import math
from typing import Any, Union

import torch
from torch import nn

from ....utils.import_utils import is_neuronx_distributed_available, is_peft_available


if is_peft_available():
    from peft.tuners.lora import Embedding as LoraEmbedding
    from peft.tuners.lora import Linear as LoraLinear
    from peft.tuners.lora import LoraLayer
    from peft.utils.integrations import gather_params_ctx
else:

    class LoraLinear:
        pass

    class LoraEmbedding:
        pass

    class LoraLayer:
        pass

    def gather_params_ctx(param):
        pass


if is_neuronx_distributed_available():
    from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear as NxDGQAQKVColumnParallelLinear
    from neuronx_distributed.parallel_layers.layers import (
        BaseParallelLinear,
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from neuronx_distributed.parallel_layers.layers import ParallelEmbedding as NxDParallelEmbedding
    from neuronx_distributed.parallel_layers.mappings import scatter_to_sequence_parallel_region
else:

    class NxDParallelEmbedding:
        def __init__(self, *args, **kwargs):
            pass

    class BaseParallelLinear:
        def __init__(self, *args, **kwargs):
            pass

    class ColumnParallelLinear:
        def __init__(self, *args, **kwargs):
            pass

    class RowParallelLinear:
        def __init__(self, *args, **kwargs):
            pass

    class NxDGQAQKVColumnParallelLinear:
        def __init__(self, *args, **kwargs):
            pass

    def scatter_to_sequence_parallel_region(*args, **kwargs):
        pass


class NeuronLoraLayer(LoraLayer):
    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, NxDGQAQKVColumnParallelLinear):
            in_features, out_features = base_layer.input_size, base_layer.output_sizes
        elif isinstance(base_layer, BaseParallelLinear):
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif isinstance(base_layer, nn.Conv2d):
            raise NotImplementedError("Conv2d is not supported for LoRA with optimum-neuron.")
        elif isinstance(base_layer, nn.Conv3d):
            raise NotImplementedError("Conv3d is not supported for LoRA with optimum-neuron.")
        elif isinstance(base_layer, NxDParallelEmbedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, nn.Conv1d):
            raise NotImplementedError("Conv1d is not supported for LoRA with optimum-neuron.")
        else:
            raise NotImplementedError(
                f"LoRA is not supported for {base_layer.__class__.__name__} with optimum-neuron."
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        # There are two cases:
        #   1. The base linear layer is a RowParallelLinear, then:
        #       - The lora A matrix needs to be a RowParallelLinear as well,
        #       - The lora B matrix does not need to be parallelized.
        #   2. The base linear layer is a ColumnParallelLinear, then:
        #       - The lora A matrix does not need to be parallelized,
        #       - The lora B matrix needs to be a ColumnParallelLinear as well.
        if isinstance(self.base_layer, RowParallelLinear):
            self.lora_A[adapter_name] = RowParallelLinear(
                self.in_features,
                r,
                bias=False,
                input_is_parallel=self.base_layer.input_is_parallel,
                sequence_parallel_enabled=self.base_layer.sequence_parallel_enabled,
                sequence_dimension=self.base_layer.sequence_dimension,
            )
            self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        else:
            self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
            self.lora_B[adapter_name] = ColumnParallelLinear(
                r,
                self.out_features,
                bias=lora_bias,
                gather_output=self.base_layer.gather_output,
                sequence_parallel_enabled=self.base_layer.sequence_parallel_enabled,
                sequence_dimension=self.base_layer.sequence_dimension,
            )

        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)


class ParallelLinear(nn.Module, NeuronLoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        NeuronLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    merge = LoraLinear.merge
    unmerge = LoraLinear.unmerge
    get_delta_weight = LoraLinear.get_delta_weight
    forward = LoraLinear.forward

    def __repr__(self):
        rep = super().__repr__()
        return "lora." + rep


class GQAQKVColumnParallelLinear(nn.Module, NeuronLoraLayer):
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        NeuronLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = NxDGQAQKVColumnParallelLinear(
            input_size=r,
            output_sizes=self.out_features,
            bias=False,
            gather_output=self.base_layer.gather_output,
            dtype=self.base_layer.dtype,
            init_method=self.base_layer.arg_init_method,
            kv_size_multiplier=self.base_layer.kv_size_multiplier,
            sequence_parallel_enabled=self.base_layer.sequence_parallel_enabled,
            fuse_qkv=self.base_layer.fuse_qkv,
        )

        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            if self.base_layer.fuse_qkv:
                nn.init.zeros_(self.lora_B[adapter_name].weight_qkv)
                if self.lora_bias[adapter_name]:
                    nn.init.zeros_(self.lora_B[adapter_name].bias_qkv)
            else:
                nn.init.zeros_(self.lora_B[adapter_name].weight_q)
                nn.init.zeros_(self.lora_B[adapter_name].weight_k)
                nn.init.zeros_(self.lora_B[adapter_name].weight_v)
                if self.lora_bias[adapter_name]:
                    nn.init.zeros_(self.lora_B[adapter_name].bias_q)
                    nn.init.zeros_(self.lora_B[adapter_name].bias_k)
                    nn.init.zeros_(self.lora_B[adapter_name].bias_v)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_dtype = x.dtype
        output_q, output_k, output_v = self.base_layer(x, *args, **kwargs)
        if not self.merged:
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                lora_dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                x = x.to(lora_A.weight.dtype)

                dropout_input = lora_A(lora_dropout(x))
                lora_q_output, lora_k_output, lora_v_output = lora_B(dropout_input)

                output_q += lora_q_output * scaling
                output_k += lora_k_output * scaling
                output_v += lora_v_output * scaling

        return output_q.to(previous_dtype), output_k.to(previous_dtype), output_v.to(previous_dtype)

    def __repr__(self):
        rep = super().__repr__()
        return "lora." + rep


class ParallelEmbedding(nn.Module, NeuronLoraLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        if lora_bias:
            # lora_bias=True is not supported (yet) for embedding layers, as they use nn.Parameter
            raise ValueError(f"lora_bias={lora_bias} is not supported for {self.__class__.__name__}.")

        super().__init__()
        NeuronLoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    update_layer = LoraEmbedding.update_layer
    dora_init = LoraEmbedding.dora_init
    merge = LoraEmbedding.merge
    unmerge = LoraEmbedding.unmerge
    get_delta_weight = LoraEmbedding.get_delta_weight
    _mixed_batch_forward = LoraEmbedding._mixed_batch_forward
    _embed = LoraEmbedding._embed

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            # If sequence parallelism is enabled, we need to scatter the input to the sequence parallel region.
            sequence_parallel_enabled = self.get_base_layer().sequence_parallel_enabled
            sequence_dimension = self.get_base_layer().sequence_dim
            if sequence_dimension is None:
                sequence_dimension = 0
            if sequence_parallel_enabled:
                if sequence_dimension == 0:
                    x = x.transpose(0, 1).contiguous()
                x = scatter_to_sequence_parallel_region(x, sequence_dimension=sequence_dimension)

            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]

                if not self.use_dora[active_adapter]:
                    after_A = self._embed(x, embedding_A)
                    result = result + (after_A @ embedding_B) * scaling
                else:
                    mag_norm_scale, dora_result = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=embedding_A,
                        lora_B=embedding_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        embed_fn=self._embed,
                    )
                    result = mag_norm_scale * result + dora_result
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self):
        rep = super().__repr__()
        return "lora." + rep


NEURON_LORA_MODULES = {
    NxDParallelEmbedding: ParallelEmbedding,
    ColumnParallelLinear: ParallelLinear,
    RowParallelLinear: ParallelLinear,
    NxDGQAQKVColumnParallelLinear: GQAQKVColumnParallelLinear,
}
