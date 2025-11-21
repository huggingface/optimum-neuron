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
from typing import Any

import torch
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear as NxDGQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers.layers import (
    BaseParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.layers import ParallelEmbedding as NxDParallelEmbedding
from neuronx_distributed.parallel_layers.mappings import scatter_to_sequence_parallel_region
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from torch import nn

from ....utils.import_utils import is_peft_available


if is_peft_available():
    from peft.tuners.lora import Embedding as LoraEmbedding
    from peft.tuners.lora import Linear as LoraLinear
    from peft.tuners.lora import LoraLayer
    from peft.tuners.lora.variants import LoraVariant
    from peft.tuners.tuners_utils import check_adapters_to_merge
    from peft.utils.integrations import gather_params_ctx
else:

    class LoraLinear:
        pass

    class LoraEmbedding:
        pass

    class LoraLayer:
        pass

    class LoraVariant:
        pass

    def gather_params_ctx(param):
        pass

    def check_adapters_to_merge(layer, adapter_names):
        return []


def use_peft_instead_of_optimum_neuron(neuron_lora_layer_method):
    """
    This decorator is used to mark methods that should not be used directly in optimum-neuron.
    Instead, the official PEFT method should be used.
    """

    def wrapper(*args, **kwargs):
        method_name = neuron_lora_layer_method.__name__
        raise NotImplementedError(
            f"Please use `peft.tuners.lora.LoraLayer.{method_name}` on the un-sharded model instead of using"
            f"`optimum.neuron.peft.lora.NeuronLoraLayer.{method_name}` on the sharded model from `optimum-neuron`."
        )

    return wrapper


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
        self.use_dora: dict[str, bool] = {}  # not actively used anymore after #2443, keep it for BC
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.lora_variant: dict[str, LoraVariant] = {}
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
        use_qalora: bool = False,
        lora_bias: bool = False,
        qalora_group_size: int = 32,
        **kwargs,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        lora_variant = self.resolve_lora_variant(
            use_dora=use_dora, use_qalora=use_qalora, qalora_group_size=qalora_group_size
        )
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

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
                tensor_model_parallel_group=self.base_layer.tensor_parallel_group,
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
                tensor_model_parallel_group=self.base_layer.tensor_parallel_group,
            )

        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_dora[adapter_name] = use_dora

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered.
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights == "orthogonal":
            with gather_params_ctx(self.get_base_layer().weight):
                self.orthogonal_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

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
        init_lora_weights: bool | str = True,
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

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> LoraVariant | None:
        if not use_dora:
            return None

        from peft.tuners.lora.variants import DoraLinearVariant

        return DoraLinearVariant()

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]

        device = lora_B.weight.device
        dtype = lora_B.weight.dtype

        # Cast to fp32 on CPU for better performance with bf16
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = lora_A.weight
        weight_B = lora_B.weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # Compute delta: B @ A * scaling
        # The result is sharded the same way as the base layer:
        # - If lora_A is RowParallelLinear: delta is sharded along input dimension
        # - If lora_B is ColumnParallelLinear: delta is sharded along output dimension
        output_tensor = (weight_B @ weight_A) * self.scaling[adapter]

        if self.fan_in_fan_out:
            output_tensor = output_tensor.transpose(0, 1)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            # Cast weights back
            lora_A.weight.data = weight_A.to(dtype)
            lora_B.weight.data = weight_B.to(dtype)

        return output_tensor

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """
        Merge the active adapter weights into the base weights.

        This works with distributed parallel linear layers (RowParallelLinear, ColumnParallelLinear).
        The merge happens on the sharded weights - each rank merges its own shard.

        Args:
            safe_merge: If True, perform merge in a copy and check for NaNs before merging.
            adapter_names: List of adapter names to merge. If None, all active adapters will be merged.
        """

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()

                if self.use_dora[active_adapter]:
                    raise NotImplementedError("DoRA is not yet supported for merge with Neuron parallel layers")

                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        lora_B = self.lora_B[active_adapter]
                        if hasattr(base_layer, "bias") and base_layer.bias is not None:
                            new_bias = base_layer.bias + lora_B.bias
                            if not torch.isfinite(new_bias).all():
                                raise ValueError(
                                    f"NaNs detected in the merged bias. The adapter {active_adapter} seems to be broken"
                                )
                            base_layer.bias.data = new_bias
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                    if self.lora_bias[active_adapter]:
                        lora_B = self.lora_B[active_adapter]
                        if hasattr(base_layer, "bias") and base_layer.bias is not None:
                            base_layer.bias.data += lora_B.bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base weights.

        This works with distributed parallel linear layers (RowParallelLinear, ColumnParallelLinear).
        The unmerge happens on the sharded weights - each rank unmerges its own shard.
        """
        if not self.merged:
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()

                if self.use_dora[active_adapter]:
                    raise NotImplementedError("DoRA is not yet supported for unmerge with Neuron parallel layers")

                delta_weight = self.get_delta_weight(active_adapter)
                base_layer.weight.data -= delta_weight

                if self.lora_bias[active_adapter]:
                    lora_B = self.lora_B[active_adapter]
                    if hasattr(base_layer, "bias") and base_layer.bias is not None:
                        base_layer.bias.data -= lora_B.bias

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if adapter_names is not None:
            raise ValueError("Mixed adapter inference is not supported with optimum-neuron.")

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                    )

            result = result.to(torch_result_dtype)

        return result

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
        init_lora_weights: bool | str = True,
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
        use_qalora: bool = False,
        lora_bias: bool = False,
        qalora_group_size: int = 32,
        **kwargs,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        lora_variant = self.resolve_lora_variant(
            use_dora=use_dora, use_qalora=use_qalora, qalora_group_size=qalora_group_size
        )
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

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
            bias=lora_bias,
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

        self.use_dora[adapter_name] = use_dora

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights == "orthogonal":
            with gather_params_ctx(self.get_base_layer().weight):
                self.orthogonal_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

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

    def get_delta_weight(self, adapter: str) -> dict[str, torch.Tensor]:
        """
        Compute the delta weights for Q, K, V for the given adapter.

        Returns a dict with keys "q", "k", "v" (or "qkv" if fused) containing the delta tensors.

        Args:
            adapter: The name of the adapter for which the delta weight should be computed.

        Returns:
            Dict mapping "q"/"k"/"v" (or "qkv") to their delta weight tensors (sharded).
        """
        lora_A = self.lora_A[adapter]
        lora_B = self.lora_B[adapter]

        device = lora_A.weight.device
        dtype = lora_A.weight.dtype

        # Cast to fp32 on CPU for better performance with fp16/bf16
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = lora_A.weight
        if cast_to_fp32:
            weight_A = weight_A.float()

        base_layer = self.get_base_layer()
        delta_weights = {}

        # Compute delta for each Q, K, V
        if base_layer.fuse_qkv:
            weight_B_qkv = lora_B.weight_qkv
            if cast_to_fp32:
                weight_B_qkv = weight_B_qkv.float()
            delta_weights["qkv"] = (weight_B_qkv @ weight_A) * self.scaling[adapter]
            if cast_to_fp32:
                delta_weights["qkv"] = delta_weights["qkv"].to(dtype=dtype)
        else:
            for key, weight_attr in [("q", "weight_q"), ("k", "weight_k"), ("v", "weight_v")]:
                weight_B = getattr(lora_B, weight_attr)
                if cast_to_fp32:
                    weight_B = weight_B.float()
                delta_weights[key] = (weight_B @ weight_A) * self.scaling[adapter]
                if cast_to_fp32:
                    delta_weights[key] = delta_weights[key].to(dtype=dtype)

        return delta_weights

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """
        Merge the active adapter weights into the base Q, K, V weights.

        This works with GQAQKVColumnParallelLinear layers.
        The merge happens on the sharded weights - each rank merges its own shard.

        Args:
            safe_merge: If True, perform merge in a copy and check for NaNs before merging.
            adapter_names: List of adapter names to merge. If None, all active adapters will be merged.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()

                if self.use_dora[active_adapter]:
                    raise NotImplementedError("DoRA is not yet supported for merge with GQA QKV layers")

                delta_weights = self.get_delta_weight(active_adapter)

                if safe_merge:
                    if base_layer.fuse_qkv:
                        orig_weight_qkv = base_layer.weight_qkv.data.clone()
                        orig_weight_qkv += delta_weights["qkv"]
                        if not torch.isfinite(orig_weight_qkv).all():
                            raise ValueError(
                                f"NaNs detected in merged QKV weights. Adapter {active_adapter} seems broken"
                            )
                        base_layer.weight_qkv.data = orig_weight_qkv
                    else:
                        for key, weight_attr in [("q", "weight_q"), ("k", "weight_k"), ("v", "weight_v")]:
                            orig_weight = getattr(base_layer, weight_attr).data.clone()
                            orig_weight += delta_weights[key]
                            if not torch.isfinite(orig_weight).all():
                                raise ValueError(
                                    f"NaNs detected in merged {key.upper()} weights. Adapter {active_adapter} seems broken"
                                )
                            getattr(base_layer, weight_attr).data = orig_weight
                else:
                    if base_layer.fuse_qkv:
                        base_layer.weight_qkv.data += delta_weights["qkv"]
                    else:
                        for key, weight_attr in [("q", "weight_q"), ("k", "weight_k"), ("v", "weight_v")]:
                            getattr(base_layer, weight_attr).data += delta_weights[key]

                # Handle bias if present
                if self.lora_bias[active_adapter]:
                    lora_B = self.lora_B[active_adapter]
                    if base_layer.fuse_qkv and hasattr(lora_B, "bias_qkv") and lora_B.bias_qkv is not None:
                        if hasattr(base_layer, "bias_qkv") and base_layer.bias_qkv is not None:
                            base_layer.bias_qkv.data += lora_B.bias_qkv
                    elif not base_layer.fuse_qkv:
                        for key, bias_attr in [("q", "bias_q"), ("k", "bias_k"), ("v", "bias_v")]:
                            if hasattr(lora_B, bias_attr) and getattr(lora_B, bias_attr) is not None:
                                if hasattr(base_layer, bias_attr) and getattr(base_layer, bias_attr) is not None:
                                    getattr(base_layer, bias_attr).data += getattr(lora_B, bias_attr)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base Q, K, V weights.

        This works with GQAQKVColumnParallelLinear layers.
        The unmerge happens on the sharded weights - each rank unmerges its own shard.
        """
        if not self.merged:
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()

                if self.use_dora[active_adapter]:
                    raise NotImplementedError("DoRA is not yet supported for unmerge with GQA QKV layers")

                delta_weights = self.get_delta_weight(active_adapter)

                if base_layer.fuse_qkv:
                    base_layer.weight_qkv.data -= delta_weights["qkv"]
                else:
                    for key, weight_attr in [("q", "weight_q"), ("k", "weight_k"), ("v", "weight_v")]:
                        getattr(base_layer, weight_attr).data -= delta_weights[key]

                # Handle bias if present
                if self.lora_bias[active_adapter]:
                    lora_B = self.lora_B[active_adapter]
                    if base_layer.fuse_qkv and hasattr(lora_B, "bias_qkv") and lora_B.bias_qkv is not None:
                        if hasattr(base_layer, "bias_qkv") and base_layer.bias_qkv is not None:
                            base_layer.bias_qkv.data -= lora_B.bias_qkv
                    elif not base_layer.fuse_qkv:
                        for key, bias_attr in [("q", "bias_q"), ("k", "bias_k"), ("v", "bias_v")]:
                            if hasattr(lora_B, bias_attr) and getattr(lora_B, bias_attr) is not None:
                                if hasattr(base_layer, bias_attr) and getattr(base_layer, bias_attr) is not None:
                                    getattr(base_layer, bias_attr).data -= getattr(lora_B, bias_attr)

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
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: bool | str = True,
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

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> LoraVariant | None:
        if not use_dora:
            return None

        from peft.tuners.lora.variants import DoraEmbeddingVariant

        return DoraEmbeddingVariant()

    update_layer = LoraEmbedding.update_layer
    _mixed_batch_forward = LoraEmbedding._mixed_batch_forward
    _embed = LoraEmbedding._embed

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # Cast to fp32 on CPU for better performance with fp16/bf16
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # Compute delta: (B @ A).T * scaling
        output_tensor = (weight_B @ weight_A).T * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)
            # Cast weights back
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        tp_size = get_tensor_model_parallel_size()
        if tp_size > 1:
            tp_rank = get_tensor_model_parallel_rank()
            base_layer = self.get_base_layer()
            # We need to slice the delta weight to match the local shard
            # The ParallelEmbedding layer pads the weight so we need to handle that
            vocab_size_per_partition = base_layer.weight.shape[0]
            start_idx = tp_rank * vocab_size_per_partition
            end_idx = start_idx + vocab_size_per_partition

            # Pad output_tensor if needed (last rank might need padding)
            if end_idx > output_tensor.shape[0]:
                pad_len = end_idx - output_tensor.shape[0]
                output_tensor = torch.nn.functional.pad(output_tensor, (0, 0, 0, pad_len))

            output_tensor = output_tensor[start_idx:end_idx, :]

        return output_tensor

    def merge(self, safe_merge: bool = False, adapter_names: list[str] | None = None) -> None:
        """
        Merge the active adapter weights into the base embedding weights.

        This works with ParallelEmbedding layers.
        The merge happens on the sharded weights - each rank merges its own shard.

        Args:
            safe_merge: If True, perform merge in a copy and check for NaNs before merging.
            adapter_names: List of adapter names to merge. If None, all active adapters will be merged.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()

                if self.use_dora[active_adapter]:
                    raise NotImplementedError("DoRA is not yet supported for merge with Neuron parallel embeddings")

                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        Unmerge all merged adapter layers from the base embedding weights.

        This works with ParallelEmbedding layers.
        The unmerge happens on the sharded weights - each rank unmerges its own shard.
        """
        if not self.merged:
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()

                if self.use_dora[active_adapter]:
                    raise NotImplementedError("DoRA is not yet supported for unmerge with Neuron parallel embeddings")

                delta_weight = self.get_delta_weight(active_adapter)
                base_layer.weight.data -= delta_weight

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
