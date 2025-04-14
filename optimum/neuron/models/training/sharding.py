from typing import Dict, List

import torch
from neuronx_distributed.parallel_layers.layers import (
    create_local_weight,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)


def slice_tensor(tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """Slice a tensor along a given axis and return a slice corresponding to the rank.
    This will round up the layer to the next multiple if there is need to pad the tensor.

    Args:
        tensor (:obj:`torch.Tensor`): The tensor to slice.
        axis (:obj:`int`): The axis along which to slice the tensor.
    """
    tp_size = get_tensor_model_parallel_size()
    axis_len = tensor.shape[axis]

    # round up to the next multiple of tp_size
    split_len = (axis_len + tp_size - 1) // tp_size
    partition_stride = 1  # assuming that is always 1
    return create_local_weight(tensor, axis, split_len, partition_stride)


def _create_local_fused_weight(tp_rank, tp_size, individual_weights, partition_dim, out_weight=None):
    weight_lists = []
    for weight in individual_weights:
        weight_list = torch.split(weight, weight.size(partition_dim) // tp_size, dim=partition_dim)[tp_rank::tp_size]
        weight_lists.append(weight_list)

    with torch.no_grad():
        return torch.cat(
            [torch.cat(weight_list, dim=partition_dim) for weight_list in weight_lists],
            dim=partition_dim,
            out=out_weight,
        )


def fuse_weights(
    keys: List[str],
    prefix: str,
    state_dict: Dict,
    partition_dim: int,
    full_fused_weights: List[torch.tensor],
    fused_name: str,
):
    """
    Fuse weights of two Linear layers the same module.

    The function can be called several times, in case the state_dict is not fully loaded.
    """
    # check if the weights are in the state_dict, and if so, it will move them to the full_fused_weights list.
    for k in keys:
        full_k = prefix + k
        if full_k in state_dict:
            full_fused_weights.append(state_dict.pop(full_k))
    # Once all the expected weights are in the full_fused_weights list, they can be fused and moved back to the
    # state_dict.
    if len(full_fused_weights) == len(keys):
        rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_size()
        state_dict[prefix + fused_name] = _create_local_fused_weight(rank, tp_size, full_fused_weights, partition_dim)
        full_fused_weights.clear()
