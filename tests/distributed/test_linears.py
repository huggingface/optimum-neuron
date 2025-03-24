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


INPUT_SIZE = 8192
OUTPUT_SIZE = 2048


class SampleModel(nn.Module):
    def __init__(self, weights_dtype):
        super().__init__()
        self.linear1 = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=False, dtype=weights_dtype)

    def forward(self, x):
        return self.linear1(x)


class ParallelSampleModel(nn.Module):
    def __init__(self, weights_dtype):
        super().__init__()
        self.linear1 = RowParallelLinear(
            INPUT_SIZE,
            OUTPUT_SIZE,
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
                # state_dict[k] = slice_tensor(v, 1)

    def forward(self, x):
        return self.linear1(x)


@torch.no_grad()
def _test_parallel_model_check(weights_dtype, inputs_dtype):
    set_seed(42)
    device = "xla"

    original_model = SampleModel(weights_dtype)
    original_model.to(device).eval()

    inputs = torch.randn(1, 2, INPUT_SIZE, dtype=inputs_dtype).to(device)
    original_outputs = original_model(inputs)
    xm.mark_step()

    parallel_model = ParallelSampleModel(weights_dtype)
    parallel_model.load_state_dict(original_model.state_dict())

    parallel_model.to(device).eval()
    parallel_outputs = parallel_model(inputs)
    outputs_match = torch.allclose(parallel_outputs.to("cpu"), original_outputs.to("cpu"), atol=1e-3)

    assert outputs_match, "Sharded model output does not match unsharded one"


def _test_parallel_linears(weights_dtype, inputs_dtype):
    run_fn = partial(_test_parallel_model_check, weights_dtype=weights_dtype, inputs_dtype=inputs_dtype)

    launch_procs(
        run_fn,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )


@is_trainium_test
@pytest.mark.xfail(reason="The output of the parallel model is not matching the original one")
def test_parallel_layers_bfloat16_inputs_bfloat16():
    _test_parallel_linears(torch.bfloat16, inputs_dtype=torch.bfloat16)

@is_trainium_test
def test_parallel_layers_bfloat16_inputs_float32():
    _test_parallel_linears(torch.bfloat16, inputs_dtype=torch.float32)


@is_trainium_test
def test_parallel_layers_float32():
    _test_parallel_linears(torch.float32, inputs_dtype=torch.float32)
