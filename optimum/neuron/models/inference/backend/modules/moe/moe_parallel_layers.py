import math
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
import torch.distributed
import torch.nn as nn
from neuronx_distributed.parallel_layers import layers, mappings, parallel_state, utils
from neuronx_distributed.parallel_layers.parallel_state import (
    get_expert_model_parallel_size,
    get_tensor_model_parallel_group,
)
from torch import Tensor
from torch.distributed import ProcessGroup


class ExpertFusedLinearWithAsyncCommunication(torch.autograd.Function):
    """Linear layer execution with asynchronous communication, specialized for
    cases where there are multiple linears that are applied to multiple inputs
    (i.e. Mixture of Experts, where there are multiple inputs, with each assigned
    to a single expert).
    In particular, this function supports parallel execution of linear layers across
    all experts, where matmuls/collectives for all experts are launched in a
    single operation (we call this "Expert Fusion").

    The implementation largely mimics LinearWithAsyncCommunication, but is modified for the 3D weights.

    notation used for shapes:
    e: number of experts
    h: input dim
    i: output dim

    shapes:
    * input/input.grad (e, ..., h)
    * output/output.grad (e, ..., i)
    * weight (e, i, h)

    NOTE: that we have inner dimensions denoted by '...', which can be an arbitrary
    number of dimensions. In general, the product of inner dimensions can be
    thought of as the number of tokens.
    Sometimes with MoE workloads it is convenient to have tokens laid out in multiple
    dimensions to facilitate tracking when they are partitioned using multiple
    parallelism dimensions.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        async_grad_allreduce: bool,
        sequence_parallel_enabled: bool,
        sequence_dimension: Optional[int] = 0,
        save_for_backward: bool = True,
        process_group: Optional[ProcessGroup] = None,
        reduce_dtype: torch.dtype = torch.float32,
    ):
        if sequence_parallel_enabled:
            raise NotImplementedError(
                "sequence parallelism (SP) is not currently supported for expert "
                "fused linear layers. If SP is in use for the model, then we "
                "currently expect SP to be exited before linear layers are applied."
            )
        # sequence_dimension parameter is unused (defined for compatibility with LinearWithAsyncCommunication)
        if input.shape[0] != weight.shape[0] and input.shape[0] > 1:
            raise RuntimeError(
                f"input and weight disagree on number of experts (first dimension). "
                f"input_shape={tuple(input.shape)}, weight_shape={tuple(weight.shape)}"
            )

        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.compute_weight_gradient = weight.requires_grad
        # TODO: Currently reduced_dtype is not used for upcasting the
        # all-reduce collective in backward, add a change to upcast
        ctx.reduce_dtype = reduce_dtype
        if process_group is None:
            process_group = get_tensor_model_parallel_group()
        ctx.process_group = process_group

        if save_for_backward:
            if ctx.compute_weight_gradient:
                ctx.save_for_backward(input, weight)
            else:
                ctx.save_for_backward(weight)

        # E: num_experts, H: input_size, I: intermediate/output_size
        # ... might refer to 1 or more dimensions, including C dimension (expert capacity)
        # input: (E, ..., H), weight: (E, H, I)
        output = torch.einsum("e...h,ehi->e...i", input, weight)

        if bias is not None:
            # Bias needs to be broadcast to the same shape as output
            bias = bias.reshape(bias.shape[0], 1, 1, bias.shape[-1])
            output += bias

        # output: (E, ..., I)
        return output

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: Sequence[Tensor]
    ) -> Tuple[
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        # grad_output: (E, ..., I)
        # input: (E, ..., H), weight: (E, H, I)
        if ctx.compute_weight_gradient:
            input, weight = ctx.saved_tensors
        else:
            weight = ctx.saved_tensors[0]
            input = None
        grad_output: torch.Tensor = grad_outputs[0]

        # grad_input: (E, ..., H)
        grad_input = torch.einsum("e...i,ehi->e...h", grad_output, weight)

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            torch.distributed.all_reduce(
                grad_input,
                group=ctx.process_group,
            )

        # if no weight gradient, immediately return
        if not ctx.compute_weight_gradient:
            return grad_input, None, None, None, None, None, None, None, None

        # grad_weight: (E, H, I)
        grad_weight = torch.einsum("e...h,e...i->ehi", input, grad_output)

        return grad_input, grad_weight, None, None, None, None, None, None, None


class ExpertFusedLinear(nn.Module):
    def _mark_expert_parallel_weights(self, iterable=None):
        """Register expert parallel parameters"""

        if get_expert_model_parallel_size() > 1:
            if iterable is None:
                iterable = self.parameters()

            for p in iterable:
                p.expert_model_parallel = True

    def _apply(self, fn, *args, **kwargs):
        """Moving parameters from cpu to device creates new parameters. to() method
        internally calls the _apply method for all the submodules, which we override
        here to make sure ep parameters are marked on device as well"""

        out = super()._apply(fn, *args, **kwargs)
        self._mark_expert_parallel_weights()
        return out

    def _save_to_state_dict(self, destination, *args, **kwargs):
        initial_states = {id(v) for v in destination.values()}
        out = super()._save_to_state_dict(destination, *args, **kwargs)
        new_states = [v for v in destination.values() if id(v) not in initial_states]
        self._mark_expert_parallel_weights(new_states)
        return out


class ExpertFusedColumnParallelLinear(layers.ColumnParallelLinear, ExpertFusedLinear):
    """Specialized linear layer for MoE, supporting column parallelism for all experts simultaneously.

    This class inherits from ColumnParallelLinear, and over-rides certain attributes and functions needed to enable
    column-parallel linear layer computation for 3D weights. The forward pass of the parent class is over-ridden
    to to support selective computations on a subset of experts.

    Bias is not currently supported for MoE.
    Sequence parallelism is handled independently of MLP computations in MoE, and therefore defaults to False.
    """

    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        stride: int = 1,
        init_method: Optional[Callable[..., Any]] = None,
        keep_master_weight: bool = False,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.num_experts = num_experts
        self._n_local_experts = utils.divide(num_experts, parallel_state.get_expert_model_parallel_size())
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            dtype=dtype,
            device=device,
            stride=stride,
            init_method=init_method,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            skip_bias_add=False,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )
        if bias:
            # Reset the partition_dim for bias to 1, as it is not correctly set in the parent class, that guesses bias
            # shape has only one dimension.
            self.bias.partition_dim = 1

        self._mark_expert_parallel_weights()

    def set_weight_and_bias_config(self):
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (
            self._n_local_experts,
            self.input_size,
            self.output_size_per_partition,
        )
        # Column parallel partitioning for each expert
        self.weight_partition_dim = 2

        if self.add_bias:
            self.bias_shape = (
                self._n_local_experts,
                self.output_size_per_partition,
            )
        else:
            self.bias_shape = None

    def _init_weight(self, weight):
        # Initialize the linear layer of each expert separately
        assert len(weight.shape) == 3
        for e in range(weight.shape[0]):
            if self.arg_init_method is None:
                torch.nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
            else:
                self.arg_init_method(weight[e])

    def forward(self, input_: torch.Tensor, expert_indices: Optional[torch.Tensor] = None, *_: Any) -> torch.Tensor:
        """If expert_indices is provided, then the computations are performed only on the specified experts.
        Otherwise, the input is passed through all experts in the layer."""

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input_
        else:
            input_parallel = mappings.copy_to_tensor_model_parallel_region(
                input_,
                process_group=self.tensor_parallel_group,
            )

        # Matrix multiply.
        weight = self.weight[expert_indices, :, :] if expert_indices is not None else self.weight
        if self.bias is not None:
            bias = self.bias[expert_indices] if expert_indices is not None else self.bias
        else:
            bias = None
        output = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
        )
        return output


class ExpertFusedRowParallelLinear(layers.RowParallelLinear, ExpertFusedLinear):
    """Specialized linear layer for MoE, supporting row parallelism for all experts simultaneously.

    This class inherits from RowParallelLinear, and over-rides certain attributes and functions needed to enable
    row-parallel linear layer computation for 3D weights. The forward pass of the parent class is over-ridden
    to optionally avoid the output all-reduce depending on the sequence parallel mode, and to support selective
    computations on a subset of experts.

    Bias is not currently supported for MoE.
    Sequence parallelism is handled independently of MLP computations in MoE, and therefore defaults to False.
    """

    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
        self,
        num_experts: int,
        input_size: int,
        output_size: int,
        bias: bool = False,
        reduce_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        stride: int = 1,
        init_method: Optional[Callable[..., Any]] = None,
        keep_master_weight: bool = False,
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.num_experts = num_experts
        self._n_local_experts = utils.divide(num_experts, parallel_state.get_expert_model_parallel_size())

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            input_is_parallel=True,
            dtype=dtype,
            device=device,
            stride=stride,
            init_method=init_method,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            skip_bias_add=False,
            reduce_output=reduce_output,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )
        self._mark_expert_parallel_weights()

    def set_weight_and_bias_config(self):
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (
            self._n_local_experts,
            self.input_size_per_partition,
            self.output_size,
        )
        # Row parallel partitioning for each expert
        self.weight_partition_dim = 1
        if self.add_bias:
            self.bias_shape = (
                self._n_local_experts,
                self.output_size,
            )
        else:
            self.bias_shape = None

    def _init_weight(self, weight):
        # Initialize the linear layer of each expert separately
        assert len(weight.shape) == 3
        for e in range(weight.shape[0]):
            if self.arg_init_method is None:
                torch.nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
            else:
                self.arg_init_method(weight[e])

    def forward(self, input_: torch.Tensor, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """If expert_indices is provided, then the computations are performed only on the specified experts.
        Otherwise, the input is passed through all experts in the layer."""

        # Matrix multiply.
        weight = self.weight[expert_indices, :, :] if expert_indices is not None else self.weight
        if self.bias is not None:
            bias = self.bias[expert_indices] if expert_indices is not None else self.bias
        else:
            bias = None
        output_parallel = self._forward_impl(
            input=input_,
            weight=weight,
            bias=bias,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
        )

        if self.reduce_output:
            output = mappings.reduce_from_tensor_model_parallel_region(
                output_parallel,
                process_group=self.tensor_parallel_group,
            )
            return output
        else:
            # Return without output all-reduce, in favor of an all-reduce or reduce-scatter after the MoE output combine.
            return output_parallel
