from functools import partial

import pytest
import torch
from torch import nn
from transformers import set_seed

from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.layers import (
        RowParallelLinear,
        create_local_weight,
    )

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xrt


class SampleModel(nn.Module):
    def __init__(self, weights_dtype, input_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size, bias=False, dtype=weights_dtype)

    def forward(self, x):
        return self.linear1(x)


class ParallelSampleModel(nn.Module):
    def __init__(self, weights_dtype, input_size, output_size):
        super().__init__()
        self.linear1 = RowParallelLinear(
            input_size,
            output_size,
            bias=False,
            dtype=weights_dtype,
            sequence_parallel_enabled=False,
        )
        # Split parallelized weights
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *_args):
        """
        Update the state dict to split the weights of the model across the world size.
        """
        # Filtering items to slice only the weights of this layer
        filtered_items = filter(lambda x: x[0].startswith(prefix), state_dict.items())
        world_size = xrt.world_size()
        for k, v in filtered_items:
            if k.endswith("linear1.weight"):
                axis_len = v.shape[1]
                split_len = (axis_len + world_size - 1) // world_size
                partition_stride = 1  # assuming that is always 1
                state_dict[k] = create_local_weight(v, 1, split_len, partition_stride)

    def forward(self, x):
        return self.linear1(x)


@torch.no_grad()
def _test_parallel_model_check(weights_dtype, inputs_dtype, input_size, output_size):
    set_seed(42)
    device = "xla"

    original_model = SampleModel(weights_dtype, input_size, output_size)
    original_model.to(device).eval()

    inputs = torch.randn(1, 2, input_size, dtype=inputs_dtype).to(device)
    original_outputs = original_model(inputs)
    xm.mark_step()

    parallel_model = ParallelSampleModel(weights_dtype, input_size, output_size)
    parallel_model.load_state_dict(original_model.state_dict())

    parallel_model.to(device).eval()
    parallel_outputs = parallel_model(inputs)
    atol = torch.finfo(inputs_dtype).resolution
    outputs_match = torch.allclose(parallel_outputs.to("cpu"), original_outputs.to("cpu"), atol=atol)

    assert outputs_match, "Sharded model output does not match unsharded one"


@is_trainium_test
@pytest.mark.parametrize(
    "weights_dtype, inputs_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float16, torch.float16),
    ],
    ids=[
        "weights=float32-inputs=float32",
        "weights=bfloat16-inputs=bfloat16",
        "weights=bfloat16-inputs=float32",
        "weights=float16-inputs=float16",
    ],
)
@pytest.mark.parametrize(
    "input_size, output_size",
    [(8192, 2048), (400, 300)],
    ids=["input_size=8192-output_size=2048", "input_size=400-output_size=300"],
)
def test_parallel_linears(weights_dtype, inputs_dtype, input_size, output_size):
    run_fn = partial(
        _test_parallel_model_check,
        weights_dtype=weights_dtype,
        inputs_dtype=inputs_dtype,
        input_size=input_size,
        output_size=output_size,
    )

    launch_procs(
        run_fn,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )
