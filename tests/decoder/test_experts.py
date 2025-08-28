from optimum.neuron.models.inference.gpt_oss.configuration_gpt_oss import GptOssConfig
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, set_seed
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssExperts, GptOssTopKRouter,
)
from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import (
    _experts_swiglu_activation,
)
from optimum.neuron.models.inference.backend.modules.moe.expert_mlps import ExpertMLPs
from optimum.neuron.models.inference.backend.modules.moe.moe_parallel_layers import ExpertFusedColumnParallelLinear, ExpertFusedRowParallelLinear
from transformers.models.mixtral.modeling_mixtral import MixtralBlockSparseTop2MLP
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear


def _set_weights(module):
    """Set the weights of the module to random values"""
    state_dict = module.state_dict()
    for key in list(state_dict.keys()):
        state_dict[key] = torch.nn.Parameter(torch.rand_like(state_dict[key]) * 0.1)
    module.load_state_dict(state_dict)


def _get_inputs(config, seq_len):
    hidden_size = config.hidden_size
    dtype = torch.float32
    # Make hidden_states between -3 and 3
    hidden_states = torch.rand(seq_len, hidden_size, dtype=dtype) * 6 - 3

    router = GptOssTopKRouter(config=config)
    _set_weights(router)
    routing_weights, indices = router(hidden_states)

    inputs = [hidden_states, routing_weights, indices, torch.tensor(seq_len)]
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
    config.hidden_size = 3
    config.num_local_experts = 4
    hidden_size = config.hidden_size
    seq_len = 2

    inputs, example_inputs = _get_inputs(config, seq_len)

    # Create a cpu layer
    cpu_module = GptOssExperts(config)
    _set_weights(cpu_module)
    cpu_module.eval()

    def wrapped_cpu_module(hidden_states, routing_weights, indices, seq_len):
        hidden_states = hidden_states.reshape(1, seq_len, hidden_size)
        ret = cpu_module(hidden_states, indices, routing_weights)
        return ret
        # return ret.reshape(seq_len, hidden_size)

    # Create a function to run the model
    neuron_model = build_module(ExpertMLPs, example_inputs, module_init_kwargs={
        "num_experts": config.num_local_experts,
        "top_k": config.num_experts_per_tok,
        "hidden_size": hidden_size,
        "intermediate_size": config.intermediate_size,
        "hidden_act": config.hidden_act,
        "glu_mlp": True,
        "capacity_factor": 1.0,
        "expert_bias": True,
        "normalize_top_k_affinities": True,
        "glu_activation_fn": _experts_swiglu_activation
    })

    state_dict = cpu_module.state_dict()
    # patch state_dict to prepend keys with "mlp_op"
    for key in list(state_dict.keys()):
        new_key = f"mlp_op.{key}"
        if new_key.endswith("_bias"):
            new_key = new_key.replace("_bias", ".bias")
        else:
            new_key += ".weight"
        state_dict[new_key] = state_dict.pop(key)

    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
    weights = [state_dict]
    neuron_model.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model
    validate_accuracy(neuron_model, inputs, cpu_callable=wrapped_cpu_module)


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
        When training is is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

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

    #hidden_size = config.hidden_size
    dtype = torch.float32
    # Make hidden_states between -3 and 3
    input_size = [seq_len, hidden_size]
    hidden_states = torch.rand(*input_size, dtype=dtype) * 6 - 3
    # repeat the hidden_states for each expert
    hidden_states = hidden_states.reshape(-1, hidden_size)  # (num_tokens, hidden_size)
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_size)

    # Create a cpu layer
    # cpu_module2 = ExpertsFusedLinear2(config.num_local_experts, config.hidden_size, 2 * config.intermediate_size, bias=True)
    cpu_module = ExpertsLinear(num_experts, hidden_size, output_size, bias=True)
    _set_weights(cpu_module)
    cpu_module.eval()
    output = cpu_module(hidden_states)

    state_dict = cpu_module.state_dict()

    inputs = [hidden_states]
    example_inputs = [torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in inputs]
    inputs = [tuple(inputs)]
    example_inputs = [tuple(example_inputs)]

    print(output.shape)

    # # Create a function to run the model
    neuron_model = build_module(expert_fused_linear_class, example_inputs, module_init_kwargs={
        "num_experts": num_experts,
        "input_size": hidden_size,
        "output_size": output_size,
        "bias": True,
        "stride": 2,
    })

    state_dict = cpu_module.state_dict()
    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
    weights = [state_dict]

    neuron_model.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model
    validate_accuracy(neuron_model, inputs, cpu_callable=cpu_module)


class NeuronGptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size

        self.gate_up_proj = ExpertFusedColumnParallelLinear(
            num_experts=self.num_experts,
            input_size=self.hidden_size,
            output_size=2 * self.expert_dim,
            bias=True,
            stride=2,
            # device=trn_config.device,
            # sequence_parallel_enabled=self.trn_config.sequence_parallel_enabled,
            # dtype=trn_config.torch_dtype,
            # tensor_model_parallel_group=self.tensor_parallel_group,
        )
        self.down_proj = ExpertFusedRowParallelLinear(
            num_experts=self.num_experts,
            input_size=self.expert_dim,
            output_size=self.hidden_size,
            bias=True,
            # reduce_output=get_expert_model_parallel_size() > 1,
            # dtype=trn_config.torch_dtype,
            # device=trn_config.device,
            # tensor_model_parallel_group=self.tensor_parallel_group,
        )

        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        """
        When training is is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size, seq_len, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        num_experts = routing_weights.shape[1]

        hidden_states = hidden_states.repeat(num_experts, 1)
        hidden_states = hidden_states.view(num_experts, -1, self.hidden_size)

        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        next_states = self.down_proj((up + 1) * glu)
        next_states = next_states.view(num_experts, batch_size, -1, self.hidden_size)
        next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        return next_states


def _get_inputs_v2(config, seq_len):
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
def test_gpt_oss_experts_v2():
    set_seed(42)
    checkpoint = "tengomucho/tiny-random-gpt-oss"
    config = GptOssConfig.from_pretrained(checkpoint)

    # Prepare the input
    seq_len = 2
    inputs, example_inputs = _get_inputs_v2(config, seq_len)

    # Create a cpu layer
    cpu_module = GptOssExperts(config)
    _set_weights(cpu_module)
    cpu_module.eval()


    # Create a function to run the model
    neuron_model = build_module(NeuronGptOssExperts, example_inputs, module_init_kwargs={
        "config": config
    })

    state_dict = cpu_module.state_dict()

    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
    state_dict["gate_up_proj.weight"] = state_dict["gate_up_proj"]
    state_dict["gate_up_proj.bias"] = state_dict["gate_up_proj_bias"]
    state_dict["down_proj.weight"] = state_dict["down_proj"]
    state_dict["down_proj.bias"] = state_dict["down_proj_bias"]

    weights = [state_dict]
    neuron_model.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model
    validate_accuracy(neuron_model, inputs, cpu_callable=cpu_module)
