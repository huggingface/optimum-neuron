import pytest
import torch
import torch.nn as nn
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from transformers import set_seed
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssExperts,
    GptOssTopKRouter,
)

from optimum.neuron.models.inference.backend.modules.moe.moe_parallel_layers import (
    ExpertFusedColumnParallelLinear,
    ExpertFusedRowParallelLinear,
)
from optimum.neuron.models.inference.gpt_oss.configuration_gpt_oss import GptOssConfig
from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import NeuronGptOssExperts
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _set_weights(module):
    """Set the weights of the module to random values"""
    state_dict = module.state_dict()
    for key in list(state_dict.keys()):
        state_dict[key] = torch.nn.Parameter(torch.rand_like(state_dict[key]) * 0.1)
    module.load_state_dict(state_dict)


class ExpertsLinear(nn.Module):
    def __init__(self, num_experts, input_size, output_size, bias=False):
        super().__init__()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.empty(self.num_experts, self.input_size, self.output_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_experts, self.output_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
        Returns:
            torch.Tensor
        """
        gate_up = torch.bmm(hidden_states, self.weight)
        if getattr(self, "bias", None) is not None:
            gate_up = gate_up + self.bias[..., None, :]
        return gate_up


@is_inferentia_test
@requires_neuronx
@torch.no_grad()
@pytest.mark.parametrize("expert_fused_linear_class", [ExpertFusedRowParallelLinear, ExpertFusedColumnParallelLinear])
def test_experts_fused_linear(expert_fused_linear_class):
    set_seed(42)

    # Prepare the input
    seq_len = 2
    num_experts = 4
    hidden_size = 6
    output_size = 10

    # hidden_size = config.hidden_size
    dtype = torch.float32
    # Make hidden_states between -3 and 3
    input_size = [seq_len, hidden_size]
    hidden_states = torch.rand(*input_size, dtype=dtype) * 6 - 3
    # repeat the hidden_states for each expert
    hidden_states = hidden_states.reshape(-1, hidden_size)  # (num_tokens, hidden_size)
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_size)

    # Create a cpu layer
    cpu_module = ExpertsLinear(num_experts, hidden_size, output_size, bias=True)
    _set_weights(cpu_module)
    cpu_module.eval()

    state_dict = cpu_module.state_dict()

    inputs = [hidden_states]
    example_inputs = [torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in inputs]
    inputs = [tuple(inputs)]
    example_inputs = [tuple(example_inputs)]

    # # Create a function to run the model
    neuron_model = build_module(
        expert_fused_linear_class,
        example_inputs,
        module_init_kwargs={
            "num_experts": num_experts,
            "input_size": hidden_size,
            "output_size": output_size,
            "bias": True,
            "stride": 2,
        },
    )

    state_dict = cpu_module.state_dict()
    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
    weights = [state_dict]

    neuron_model.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model
    validate_accuracy(neuron_model, inputs, cpu_callable=cpu_module)


def _get_inputs(config, seq_len):
    hidden_size = config.hidden_size
    dtype = torch.float32
    # Make hidden_states between -3 and 3
    hidden_states = torch.rand(seq_len, hidden_size, dtype=dtype) * 6 - 3

    router = GptOssTopKRouter(config=config)
    _set_weights(router)
    routing_weights, indices = router(hidden_states)

    inputs = [hidden_states, indices, routing_weights]
    example_inputs = [torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in inputs]
    inputs = [tuple(inputs)]
    example_inputs = [tuple(example_inputs)]
    return inputs, example_inputs


@is_inferentia_test
@requires_neuronx
@torch.no_grad()
def test_gpt_oss_experts():
    set_seed(42)
    checkpoint = "tengomucho/tiny-random-gpt-oss"
    config = GptOssConfig.from_pretrained(checkpoint)

    # Prepare the input
    seq_len = 2
    inputs, example_inputs = _get_inputs(config, seq_len)

    # Create a cpu layer
    cpu_module = GptOssExperts(config)
    _set_weights(cpu_module)
    cpu_module.eval()

    # Create a function to run the model
    neuron_model = build_module(NeuronGptOssExperts, example_inputs, module_init_kwargs={"config": config})

    state_dict = cpu_module.state_dict()

    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
    # Adapt state dict to the neuron model
    state_dict["gate_up_proj.weight"] = state_dict["gate_up_proj"]
    state_dict["gate_up_proj.bias"] = state_dict["gate_up_proj_bias"]
    state_dict["down_proj.weight"] = state_dict["down_proj"]
    state_dict["down_proj.bias"] = state_dict["down_proj_bias"]

    weights = [state_dict]
    neuron_model.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model
    validate_accuracy(neuron_model, inputs, cpu_callable=cpu_module)
