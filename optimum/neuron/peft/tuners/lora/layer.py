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
from typing import Union

from torch import nn

from ....utils.import_utils import is_neuronx_distributed_available, is_peft_available


if is_peft_available():
    from peft.tuners.lora import Linear as LoraLinear
    from peft.tuners.lora import LoraLayer
    from peft.utils.integrations import gather_params_ctx
else:

    class LoraLayer:
        pass

    def gather_params_ctx(param):
        pass


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        ParallelEmbedding,
        RowParallelLinear,
    )
else:

    class ParallelEmbedding:
        pass

    class ColumnParallelLinear:
        pass

    class RowParallelLinear:
        pass


class NeuronLoraLayer(LoraLayer):
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


class LoraParallelEmbedding(nn.Module, NeuronLoraLayer):
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


NEURON_LORA_MODULES = {
    # TODO: handle embeddings
    # ParallelEmbedding: LoraParallelEmbedding,
    ColumnParallelLinear: ParallelLinear,
    RowParallelLinear: ParallelLinear,
}
