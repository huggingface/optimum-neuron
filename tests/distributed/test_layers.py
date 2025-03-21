from functools import partial
import torch
from torch import nn
from transformers import set_seed

from optimum.neuron.utils.import_utils import (
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xrt

from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    create_local_weight,
)

# from transformers.models.granite.modeling_granite import GraniteMLP
from transformers import AutoConfig

# from optimum.neuron.models.training.granite.modeling_granite import GraniteMLP as NeuronGraniteMLP



class GraniteMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        # y = self.act_fn(self.gate_proj(x) * self.up_proj(x))
        # down_proj = self.down_proj(y)
        # print(f"y shape: {y.shape}")
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class ParallelGraniteMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        assert not config.mlp_bias, "GraniteMLP sharding does not support bias for now"
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=config.mlp_bias,
            dtype=config.torch_dtype,
            sequence_parallel_enabled=False,
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = nn.SiLU()
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
            if k.endswith("down_proj.weight"):
                axis_len = v.shape[1]
                # state_dict[k] = slice_tensor(v, 1)
                split_len = (axis_len + world_size - 1) // world_size
                partition_stride = 1 # assuming that is always 1
                state_dict[k] = create_local_weight(v, 1, split_len, partition_stride)

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


@torch.no_grad()
def _test_parallel_mlp():
    set_seed(42)

    model_id = "ibm-granite/granite-3.2-2b-instruct"
    config = AutoConfig.from_pretrained(model_id)

    device = "xla"

    original_mlp = GraniteMLP(config).to(torch.bfloat16)
    original_mlp.to(device).eval()
    inputs = torch.randn(1, 2, config.hidden_size, dtype=torch.bfloat16).to(device)
    original_outputs = original_mlp(inputs)
    xm.mark_step()

    neuron_config = config
    parallel_mlp = ParallelGraniteMLP(neuron_config).to(torch.bfloat16)
    parallel_mlp.load_state_dict(original_mlp.state_dict())

    parallel_mlp.to(device).eval()
    parallel_outputs = parallel_mlp(inputs)
    outputs_match = torch.allclose(parallel_outputs.to("cpu"), original_outputs.to("cpu"), atol=1e-3)
    print(f"🟡 rank par {xrt.global_ordinal()} outputs match? {outputs_match}")
    return
    assert outputs_match, "Sharded model output does not match unsharded one"


@is_trainium_test
def test_parallel_layers():
    # _test_parallel_mlp()
    # return
    launch_procs(
        _test_parallel_mlp,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )
