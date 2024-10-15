import enum
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.pad import get_number_of_extra_heads
from neuronx_distributed.quantization.quantization_layers import (
    BaseQuantizeParallelLinear,
)


class GQA(enum.Enum):
    # This transforms a GQA attention mechanism into a traditional MHA mechanism
    # by replicating the K/V heads to evenly match the corresponding Q heads.
    # This consumes more memory than would otherwise be used with other sharding
    # mechanisms but works in all cases.
    # Example:
    # tp_degree = 32
    # num_attention_heads: 56 -> 64
    # num_kev_value_heads: 8  -> 64
    # | K1 K1 | K2 K2 | ... | K7 K7| Pad Pad | ... | Pad Pad |
    # | Q1 Q2 | Q3 Q4 | ... | Q55 Q56 | Pad Pad | ... | Pad Pad |
    CONVERT_TO_MHA = "convert-to-mha"

    # This transforms a GQA attention mechanism such that there is extactly
    # one K/V head per tp_degree through replication e.g. 8 K/V heads with
    # tp_degree=32 results in 32 K/V heads. This is more memory efficient but
    # does not work for all configurations. Q heads are padded interleaved
    # to retain correct alignment between Q and K/V heads.
    # Example:
    # tp_degree = 32
    # num_attention_heads: 56 -> 64
    # num_kev_value_heads: 8  -> 32
    # | K1    | K1    | K1    | K1     | K2    | ...
    # | Q1 Q2 | Q3 Q4 | Q5 Q6 | Q7 Pad | Q8 Q9 | ...
    REPLICATE_TO_TP_DEGREE = "replicate-to-tp-degree"


def determine_sharding_strategy(
    tp_degree: int, source_key_value_heads: int, desired_sharding_strategy: Optional[GQA] = None
) -> GQA:
    sharding_strategy = desired_sharding_strategy if desired_sharding_strategy else GQA.REPLICATE_TO_TP_DEGREE

    if sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE and (tp_degree % source_key_value_heads != 0):
        sharding_strategy = GQA.CONVERT_TO_MHA

    return sharding_strategy


def get_shardable_head_counts(
    tp_degree: int, num_attention_heads: int, num_key_value_heads: int, sharding_strategy: GQA
) -> Tuple[int, int]:
    # Pad attention heads
    updated_num_attention_heads = num_attention_heads + get_number_of_extra_heads(num_attention_heads, tp_degree)

    # Replicate and pad K/V heads
    updated_num_key_value_heads = num_key_value_heads
    if num_attention_heads == num_key_value_heads:  # MHA
        updated_num_key_value_heads = updated_num_attention_heads
    else:  # GQA / MQA
        if (num_key_value_heads < tp_degree) or (num_key_value_heads % tp_degree != 0):
            if sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
                assert (
                    tp_degree % num_key_value_heads == 0
                ), "GQA.REPLICATE_TO_TP_DEGREE requires tp_degree to be divisible by num_key_value_heads"
                updated_num_key_value_heads = tp_degree
            elif sharding_strategy == GQA.CONVERT_TO_MHA:
                updated_num_key_value_heads = updated_num_attention_heads

    return updated_num_attention_heads, updated_num_key_value_heads


def is_per_channel(scale: torch.Tensor) -> bool:
    """See if the scale is per channel"""
    if scale.shape == (1,):
        return False
    return True


def get_tensor_per_channel_scale_axis(scale: torch.Tensor) -> int:
    """Get the channel axis for the per channel scale"""
    scale_shape = scale.shape
    # Only one dimension would have scale values
    for i, dim_length in enumerate(scale_shape):
        if dim_length > 1:
            return i
    raise RuntimeError(f"Cannot get channel axis for the scale: {scale}")


def should_pad_scale(tensor_scale: torch.Tensor, pad_dim: int) -> bool:
    """Should scale be padded"""
    if (
        (tensor_scale is not None)
        and (is_per_channel(tensor_scale))
        and (get_tensor_per_channel_scale_axis(tensor_scale) == pad_dim)
    ):
        return True
    return False


def verify_scale_dimension(tensor: torch.Tensor, tensor_scale: torch.Tensor):
    channel_axis = get_tensor_per_channel_scale_axis(scale=tensor_scale)
    assert tensor_scale.shape[channel_axis] == tensor.shape[channel_axis]


def maybe_pad_interleaved(
    tensor,
    pad_dim: int,
    source_heads: int,
    target_heads: int,
    source_group_size: int,
    tensor_scale: torch.Tensor = None,
):
    tensor = _maybe_pad_interleaved(tensor, pad_dim, source_heads, target_heads, source_group_size)
    if should_pad_scale(tensor_scale=tensor_scale, pad_dim=pad_dim):
        tensor_scale = _maybe_pad_interleaved(tensor_scale, pad_dim, source_heads, target_heads, source_group_size)

    return tensor, tensor_scale


def _maybe_pad_interleaved(tensor, pad_dim: int, source_heads: int, target_heads: int, source_group_size: int):
    if tensor is None:
        return tensor
    shape = tensor.shape[:pad_dim] + (source_heads, tensor.shape[pad_dim] // source_heads) + tensor.shape[pad_dim + 1 :]
    tensor = tensor.view(shape)

    splits = torch.split(tensor, source_group_size, dim=pad_dim)

    pad_size = list(splits[0].size())
    pad_size[pad_dim] = (target_heads - source_heads) // (source_heads // source_group_size)
    pads = [torch.zeros(pad_size, dtype=tensor.dtype)] * len(splits)

    interleaved = [t for pair in zip(splits, pads) for t in pair]
    tensor = torch.cat(interleaved, dim=pad_dim)

    shape = tensor.shape[:pad_dim] + (tensor.shape[pad_dim] * tensor.shape[pad_dim + 1],) + tensor.shape[pad_dim + 2 :]
    return tensor.view(shape)


def maybe_pad_tail(tensor, source_heads: int, target_heads: int, pad_dim: int, tensor_scale=None):
    tensor = _maybe_pad_tail(tensor, source_heads, target_heads, pad_dim)
    if should_pad_scale(tensor_scale=tensor_scale, pad_dim=pad_dim):
        tensor_scale = _maybe_pad_tail(tensor_scale, source_heads, target_heads, pad_dim)
    return tensor, tensor_scale


def _maybe_pad_tail(tensor, source_heads: int, target_heads: int, pad_dim: int):
    if tensor is None:
        return tensor
    size_to_pad = int((tensor.shape[pad_dim] // source_heads) * target_heads - tensor.shape[pad_dim])

    dims_after_pad_dim = len(tensor.size()) - pad_dim
    pad_length = dims_after_pad_dim * 2
    pad = (0,) * (pad_length - 1) + (size_to_pad,)

    return F.pad(tensor, pad)


def replicate_kv(tensor, source_heads: int, repeats: int, head_dim=0, tensor_scale=None):
    tensor = _replicate_kv(tensor=tensor, source_heads=source_heads, repeats=repeats, head_dim=head_dim)
    if should_pad_scale(tensor_scale=tensor_scale, pad_dim=head_dim):
        tensor_scale = _replicate_kv(tensor=tensor_scale, source_heads=source_heads, repeats=repeats, head_dim=head_dim)
    return tensor, tensor_scale


def _replicate_kv(tensor, source_heads: int, repeats: int, head_dim=0):
    if tensor is None:
        return tensor
    shape = (
        tensor.shape[:head_dim] + (source_heads, tensor.shape[head_dim] // source_heads) + tensor.shape[head_dim + 1 :]
    )
    tensor = tensor.view(shape)
    tensor = torch.repeat_interleave(tensor, repeats=repeats, dim=head_dim)
    shape = (
        tensor.shape[:head_dim] + (tensor.shape[head_dim] * tensor.shape[head_dim + 1],) + tensor.shape[head_dim + 2 :]
    )
    return tensor.view(shape)


class BaseGroupQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.tp_degree = tp_degree
        self.head_dim = head_dim
        self.dtype = dtype
        self.bias = bias
        self._src_num_attention_heads = num_attention_heads
        self._src_num_key_value_heads = num_key_value_heads

        self.sharding_strategy = determine_sharding_strategy(
            tp_degree, self._src_num_key_value_heads, desired_sharding_strategy=desired_sharding_strategy
        )
        self.num_attention_heads, self.num_key_value_heads = get_shardable_head_counts(
            tp_degree, self._src_num_attention_heads, self._src_num_key_value_heads, self.sharding_strategy
        )

    def get_sharding_strategy(self) -> GQA:
        return self.sharding_strategy

    def get_num_attention_heads(self) -> int:
        return self.num_attention_heads

    def get_num_key_value_heads(self) -> int:
        return self.num_key_value_heads

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        raise NotImplementedError

    def replace_prefixes(self, old_prefix, new_prefix, model_state_dict):
        old_keys = []
        new_keys = []
        for key in model_state_dict.keys():
            if old_prefix in key:
                new_key = key.replace(old_prefix, new_prefix)
                new_keys.append(new_key)
                old_keys.append(key)

        for key_index in range(len(old_keys)):
            model_state_dict[new_keys[key_index]] = model_state_dict.pop(old_keys[key_index])


class GroupQueryAttention_QKV(BaseGroupQueryAttention):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
        gather_output: bool = True,
        fused_qkv: bool = False,
        clip_qkv: Optional[float] = None,
    ):
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            tp_degree=tp_degree,
            dtype=dtype,
            bias=bias,
            desired_sharding_strategy=desired_sharding_strategy,
        )
        if fused_qkv and gather_output:
            raise ValueError(
                "Gathering states followed by fused qkv is not allowed as it has a different weight sharding scheme."
            )

        self.gather_output = gather_output
        self.fused_qkv = fused_qkv
        self.clip_qkv = clip_qkv

        if parallel_state.model_parallel_is_initialized():
            if self.fused_qkv:
                self.Wqkv = ColumnParallelLinear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                )
                # Set heads info as weight parameter attributes to be used in weights sharding
                setattr(self.Wqkv.weight, "fused_qkv", True)
                setattr(self.Wqkv.weight, "num_attention_heads", self.num_attention_heads)
                setattr(self.Wqkv.weight, "num_key_value_heads", self.num_key_value_heads)
                setattr(self.Wqkv.weight, "head_dim", self.head_dim)
            else:
                self.q_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_attention_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                )
                self.k_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                )
                self.v_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.num_key_value_heads * self.head_dim,
                    bias=self.bias,
                    gather_output=self.gather_output,
                    dtype=dtype,
                )
        else:
            if self.fused_qkv:
                self.Wqkv = nn.Linear(
                    self.hidden_size,
                    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
                    bias=self.bias,
                )
            else:
                self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=self.bias)
                self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias)
                self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.bias)

    def forward(self, hidden_states: torch.Tensor):
        if self.fused_qkv:
            QKV = self.Wqkv(hidden_states)
            if self.clip_qkv is not None:
                QKV = QKV.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            # torch.split has accuracy issue and leads to more reshapes in hlo.
            # Using torch.tensor_split here. NAPP-3145
            Q, K, V = torch.tensor_split(
                QKV,
                (
                    self.num_attention_heads * self.head_dim // self.tp_degree,
                    (self.num_attention_heads + self.num_key_value_heads) * self.head_dim // self.tp_degree,
                ),
                dim=2,
            )
        else:
            Q = self.q_proj(hidden_states)
            K = self.k_proj(hidden_states)
            V = self.v_proj(hidden_states)
            if self.clip_qkv is not None:
                Q = Q.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                K = K.clamp(min=-self.clip_qkv, max=self.clip_qkv)
                V = V.clamp(min=-self.clip_qkv, max=self.clip_qkv)
        return Q, K, V

    def get_weight(
        self, prefix: str, layer: torch.nn.Module, layer_name, model_state_dict: dict
    ) -> Tuple[torch.Tensor]:
        if hasattr(layer, "get_weight_from_state_dict"):
            weight = layer.get_weight_from_state_dict(prefix=f"{prefix}.{layer_name}.", state_dict=model_state_dict)
            if isinstance(layer, BaseQuantizeParallelLinear):
                scale = layer.get_scale_from_state_dict(prefix=f"{prefix}.{layer_name}.", state_dict=model_state_dict)
            else:
                scale = None
        else:
            weight = model_state_dict[f"{prefix}.{layer_name}.weight"]
            if isinstance(layer, BaseQuantizeParallelLinear):
                scale = model_state_dict[f"{prefix}.{layer_name}.scale"]
            else:
                scale = None
        return weight, scale

    def get_bias(
        self, prefix: str, layer: torch.nn.Module, layer_name: str, model_state_dict: dict
    ) -> Tuple[torch.Tensor]:
        if hasattr(layer, "get_bias_from_state_dict"):
            bias = layer.get_bias_from_state_dict(prefix=f"{prefix}.{layer_name}.", state_dict=model_state_dict)
        else:
            bias = model_state_dict.get(f"{prefix}.{layer_name}.bias")
        return bias

    def set_weight(
        self,
        tensor: torch.Tensor,
        prefix: str,
        layer: torch.nn.Module,
        layer_name,
        model_state_dict: dict,
        scale: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        # TODO: set weight to state dict support is pending.
        model_state_dict[f"{prefix}.{layer_name}.weight"] = tensor
        if scale is not None:
            model_state_dict[f"{prefix}.{layer_name}.scale"] = scale
            verify_scale_dimension(tensor=tensor, tensor_scale=scale)

    def set_bias(
        self, tensor: torch.Tensor, prefix: str, layer: torch.nn.Module, layer_name: str, model_state_dict: dict
    ) -> Tuple[torch.Tensor]:
        if hasattr(layer, "set_bias_to_state_dict"):
            layer.set_bias_to_state_dict(prefix=f"{prefix}.{layer_name}.", tensor=tensor, state_dict=model_state_dict)
        else:
            model_state_dict[f"{prefix}.{layer_name}.bias"] = tensor

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        prefix_parts = prefix.split(".")
        prefix = ".".join(prefix_parts[:-1])
        hf_prefix = ".".join(prefix_parts[:-2])
        if self.fused_qkv:
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.Wqkv", new_prefix=f"{prefix}.Wqkv", model_state_dict=model_state_dict
            )
            qkv_weight, _ = self.get_weight(
                prefix=prefix, layer=self.Wqkv, layer_name="Wqkv", model_state_dict=model_state_dict
            )
            q_proj_weight, k_proj_weight, v_proj_weight = qkv_weight.split(
                [
                    self._src_num_attention_heads * self.head_dim,
                    self._src_num_key_value_heads * self.head_dim,
                    self._src_num_key_value_heads * self.head_dim,
                ],
                dim=0,
            )
            q_proj_scale, k_proj_scale, v_proj_scale = None, None, None
            qkv_bias = self.get_bias(
                prefix=prefix, layer=self.Wqkv, layer_name="Wqkv", model_state_dict=model_state_dict
            )
            if qkv_bias is not None:
                q_proj_bias, k_proj_bias, v_proj_bias = qkv_bias.split(
                    [
                        self._src_num_attention_heads * self.head_dim,
                        self._src_num_key_value_heads * self.head_dim,
                        self._src_num_key_value_heads * self.head_dim,
                    ],
                    dim=0,
                )
            else:
                q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
        else:
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.q_proj", new_prefix=f"{prefix}.q_proj", model_state_dict=model_state_dict
            )
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.k_proj", new_prefix=f"{prefix}.k_proj", model_state_dict=model_state_dict
            )
            self.replace_prefixes(
                old_prefix=f"{hf_prefix}.v_proj", new_prefix=f"{prefix}.v_proj", model_state_dict=model_state_dict
            )

            q_proj_weight, q_proj_scale = self.get_weight(
                prefix=prefix, layer=self.q_proj, layer_name="q_proj", model_state_dict=model_state_dict
            )
            k_proj_weight, k_proj_scale = self.get_weight(
                prefix=prefix, layer=self.k_proj, layer_name="k_proj", model_state_dict=model_state_dict
            )
            v_proj_weight, v_proj_scale = self.get_weight(
                prefix=prefix, layer=self.v_proj, layer_name="v_proj", model_state_dict=model_state_dict
            )

            q_proj_bias = self.get_bias(
                prefix=prefix, layer=self.q_proj, layer_name="q_proj", model_state_dict=model_state_dict
            )
            k_proj_bias = self.get_bias(
                prefix=prefix, layer=self.k_proj, layer_name="k_proj", model_state_dict=model_state_dict
            )
            v_proj_bias = self.get_bias(
                prefix=prefix, layer=self.v_proj, layer_name="v_proj", model_state_dict=model_state_dict
            )

        if self.num_key_value_heads != self._src_num_key_value_heads:
            if self.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
                repeats = self.tp_degree // self._src_num_key_value_heads
            elif self.sharding_strategy == GQA.CONVERT_TO_MHA:
                repeats = self._src_num_attention_heads // self._src_num_key_value_heads
            k_proj_weight, k_proj_scale = replicate_kv(
                k_proj_weight,
                source_heads=self._src_num_key_value_heads,
                repeats=repeats,
                head_dim=0,
                tensor_scale=k_proj_scale,
            )
            k_proj_bias, _ = replicate_kv(
                k_proj_bias, source_heads=self._src_num_key_value_heads, repeats=repeats, head_dim=0
            )
            v_proj_weight, v_proj_scale = replicate_kv(
                v_proj_weight,
                source_heads=self._src_num_key_value_heads,
                repeats=repeats,
                head_dim=0,
                tensor_scale=v_proj_scale,
            )
            v_proj_bias, _ = replicate_kv(
                v_proj_bias, source_heads=self._src_num_key_value_heads, repeats=repeats, head_dim=0
            )

        if self.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
            q_proj_weight, q_proj_scale = maybe_pad_interleaved(
                q_proj_weight,
                pad_dim=0,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                source_group_size=self._src_num_attention_heads // self._src_num_key_value_heads,
                tensor_scale=q_proj_scale,
            )
            q_proj_bias, _ = maybe_pad_interleaved(
                q_proj_bias,
                pad_dim=0,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                source_group_size=self._src_num_attention_heads // self._src_num_key_value_heads,
            )

        if self.sharding_strategy == GQA.CONVERT_TO_MHA:
            q_proj_weight, q_proj_scale = maybe_pad_tail(
                q_proj_weight,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                pad_dim=0,
                tensor_scale=q_proj_scale,
            )
            q_proj_bias, _ = maybe_pad_tail(
                q_proj_bias,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                pad_dim=0,
            )
            k_proj_weight, k_proj_scale = maybe_pad_tail(
                k_proj_weight,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
                tensor_scale=k_proj_scale,
            )
            k_proj_bias, _ = maybe_pad_tail(
                k_proj_bias,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
            )
            v_proj_weight, v_proj_scale = maybe_pad_tail(
                v_proj_weight,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
                tensor_scale=v_proj_scale,
            )
            v_proj_bias, _ = maybe_pad_tail(
                v_proj_bias,
                source_heads=self._src_num_key_value_heads,
                target_heads=self.num_key_value_heads,
                pad_dim=0,
            )

        if self.fused_qkv:
            qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
            self.set_weight(
                tensor=qkv_weight,
                prefix=prefix,
                layer=self.Wqkv,
                layer_name="Wqkv",
                model_state_dict=model_state_dict,
            )
            if self.bias:
                qkv_bias = torch.cat([q_proj_bias, k_proj_bias, v_proj_bias], dim=0)
                self.set_bias(
                    tensor=qkv_bias,
                    prefix=prefix,
                    layer=self.Wqkv,
                    layer_name="Wqkv",
                    model_state_dict=model_state_dict,
                )
        else:
            self.set_weight(
                tensor=q_proj_weight,
                prefix=prefix,
                layer=self.q_proj,
                layer_name="q_proj",
                model_state_dict=model_state_dict,
                scale=q_proj_scale,
            )
            self.set_weight(
                tensor=k_proj_weight,
                prefix=prefix,
                layer=self.k_proj,
                layer_name="k_proj",
                model_state_dict=model_state_dict,
                scale=k_proj_scale,
            )
            self.set_weight(
                tensor=v_proj_weight,
                prefix=prefix,
                layer=self.v_proj,
                layer_name="v_proj",
                model_state_dict=model_state_dict,
                scale=v_proj_scale,
            )

            if self.bias:
                self.set_bias(
                    tensor=q_proj_bias,
                    prefix=prefix,
                    layer=self.q_proj,
                    layer_name="q_proj",
                    model_state_dict=model_state_dict,
                )
                self.set_bias(
                    tensor=k_proj_bias,
                    prefix=prefix,
                    layer=self.k_proj,
                    layer_name="k_proj",
                    model_state_dict=model_state_dict,
                )
                self.set_bias(
                    tensor=v_proj_bias,
                    prefix=prefix,
                    layer=self.v_proj,
                    layer_name="v_proj",
                    model_state_dict=model_state_dict,
                )

        return True


class GroupQueryAttention_O(BaseGroupQueryAttention):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        tp_degree: int = 1,
        dtype: torch.dtype = torch.float32,
        bias: bool = False,
        desired_sharding_strategy: Optional[GQA] = None,
        input_is_parallel: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            tp_degree=tp_degree,
            dtype=dtype,
            bias=bias,
            desired_sharding_strategy=desired_sharding_strategy,
        )

        self.input_is_parallel = input_is_parallel

        if parallel_state.model_parallel_is_initialized():
            self.o_proj = RowParallelLinear(
                self.num_attention_heads * self.head_dim,
                self.hidden_size,
                bias=self.bias,
                input_is_parallel=self.input_is_parallel,
                dtype=self.dtype,
            )
        else:
            self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=self.bias)

    def forward(self, attention_output: torch.Tensor):
        o = self.o_proj(attention_output)
        return o

    def preshard_hook(self, model_state_dict: dict, prefix: str) -> bool:
        prefix_parts = prefix.split(".")
        prefix = ".".join(prefix_parts[:-1])
        hf_prefix = ".".join(prefix_parts[:-2])

        self.replace_prefixes(
            old_prefix=f"{hf_prefix}.o_proj", new_prefix=f"{prefix}.o_proj", model_state_dict=model_state_dict
        )
        o_proj_weight = model_state_dict[f"{prefix}.o_proj.weight"]
        o_proj_scale = model_state_dict.get(f"{prefix}.o_proj.scale", None)

        if self.sharding_strategy == GQA.REPLICATE_TO_TP_DEGREE:
            o_proj_weight, o_proj_scale = maybe_pad_interleaved(
                o_proj_weight,
                pad_dim=1,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                source_group_size=self._src_num_attention_heads // self._src_num_key_value_heads,
                tensor_scale=o_proj_scale,
            )

        if self.sharding_strategy == GQA.CONVERT_TO_MHA:
            o_proj_weight, o_proj_scale = maybe_pad_tail(
                o_proj_weight,
                source_heads=self._src_num_attention_heads,
                target_heads=self.num_attention_heads,
                pad_dim=1,
                tensor_scale=o_proj_scale,
            )

        model_state_dict[f"{prefix}.o_proj.weight"] = o_proj_weight
        if o_proj_scale is not None:
            model_state_dict[f"{prefix}.o_proj.scale"] = o_proj_scale
            verify_scale_dimension(tensor=o_proj_weight, tensor_scale=o_proj_scale)

        return True
