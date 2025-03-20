import torch
from transformers import set_seed

from optimum.neuron.models.training.granite.modeling_granite import (
    slice_tensor,
)
from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xrt
if is_neuronx_distributed_available():
    pass
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)


OUTPUT_SIZE = 2048
INPUT_SIZE = 8192

class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(INPUT_SIZE, OUTPUT_SIZE, False, dtype=torch.bfloat16)

    def forward(self, x):
        x = self.linear1(x)
        return x


class ParallelSampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = RowParallelLinear(INPUT_SIZE, OUTPUT_SIZE, False, dtype=torch.bfloat16)

        # Split parallelized weights
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, _prefix, *_args):
        """
        Update the state dict to split the weights of the model across the world size.
        """
        for k, v in state_dict.items():
            if k == "linear1.weight":
                state_dict[k] = slice_tensor(v, 1)

    def forward(self, x):
        x = self.linear1(x)
        return x


@torch.no_grad()
def _test_parallel_linears():
    set_seed(42)
    original_model = SampleModel()
    original_model.to("xla").eval()
    inputs = torch.randn(2, INPUT_SIZE).to("xla")
    original_outputs = original_model(inputs)
    xm.mark_step()
    print(f"🟡 rank ori {xrt.global_ordinal()} ", original_outputs)

    parallel_model = ParallelSampleModel()
    parallel_model.load_state_dict(original_model.state_dict())

    parallel_model.to("xla").eval()
    parallel_outputs = parallel_model(inputs)
    print(f"🟡 rank par {xrt.global_ordinal()} ", parallel_outputs)
    outputs_match = torch.allclose(parallel_outputs.to("cpu"), original_outputs.to("cpu"), atol=1e-3)
    assert outputs_match, "Sharded model output does not match unsharded one"


@is_trainium_test
def test_parallel_linears():
    launch_procs(
        _test_parallel_linears,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )
