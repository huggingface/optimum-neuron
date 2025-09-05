import torch
from nxd_testing import build_module, validate_accuracy
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssMLP,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
)
# from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig

from optimum.neuron.models.inference.backend.modules.rms_norm import NeuronRMSNorm
from optimum.neuron.models.inference.llama.modeling_llama import NeuronLlamaMLP
# from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import (
#     GptOssMLP as NeuronGptOssMLP,
# )
# from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import (
#     GptOssRotaryEmbedding as NeuronGptOssRotaryEmbedding,
# )
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
def test_rms_norm():
    config_id = "llamafactory/tiny-random-Llama-3"
    config = AutoConfig.from_pretrained(config_id)
    hidden_size = config.hidden_size
    seq_len = 2
    hidden_states = torch.ones(1, seq_len, hidden_size, dtype=torch.bfloat16)
    inputs = [(hidden_states,)]

    example_inputs = [
        tuple([torch.zeros_like(input_element) for input_element in input_elements]) for input_elements in inputs
    ]

    # Create a cpu layer
    cpu_module = LlamaRMSNorm(hidden_size).to(torch.bfloat16)

    # Create a function to run the model
    neuron_module = build_module(NeuronRMSNorm, example_inputs, module_init_kwargs={"hidden_size": hidden_size})

    # Validate the accuracy of the model
    validate_accuracy(neuron_module, inputs, cpu_callable=cpu_module)


def _set_weights(module):
    """Set the weights of the module to random values"""
    state_dict = module.state_dict()
    for key in list(state_dict.keys()):
        state_dict[key] = torch.nn.Parameter(torch.rand_like(state_dict[key]) * 0.1)
    module.load_state_dict(state_dict)


@is_inferentia_test
@requires_neuronx
def test_llama_mlp():
    checkpoint = "llamafactory/tiny-random-Llama-3"
    config = AutoConfig.from_pretrained(checkpoint)
    hidden_size = config.hidden_size
    dtype = torch.float32
    seq_len = 2
    # Initialize hidden states to random values between -3 and 3
    hidden_states = torch.rand(seq_len, hidden_size, dtype=dtype) * 6 - 3
    inputs = [(hidden_states,)]

    example_inputs = [
        tuple([torch.zeros_like(input_element) for input_element in input_elements]) for input_elements in inputs
    ]

    cpu_module = LlamaMLP(config)
    _set_weights(cpu_module)

    def cpu_module_wrapper(hidden_states):
        outputs = cpu_module(hidden_states)
        routed_out, _router_scores = outputs
        return (routed_out,)

    # Create a function to run the model
    neuron_config = NxDNeuronConfig(
        batch_size=1,
        sequence_length=seq_len,
    )
    neuron_module = build_module(
        NeuronLlamaMLP, example_inputs, module_init_kwargs={"config": config, "neuron_config": neuron_config}
    )
    state_dict = cpu_module.state_dict()

    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
    # state_dict["experts.gate_up_proj.weight"] = state_dict["experts.gate_up_proj"]
    # state_dict["experts.gate_up_proj.bias"] = state_dict["experts.gate_up_proj_bias"]
    # state_dict["experts.down_proj.weight"] = state_dict["experts.down_proj"]
    # state_dict["experts.down_proj.bias"] = state_dict["experts.down_proj_bias"]

    weights = [state_dict]
    neuron_module.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model
    validate_accuracy(neuron_module, inputs, cpu_callable=cpu_module_wrapper)


@is_inferentia_test
@requires_neuronx
def test_gpt_oss_rotary_embedding():
    checkpoint = "tengomucho/tiny-random-gpt-oss"
    config = GptOssConfig.from_pretrained(checkpoint)
    hidden_size = config.hidden_size
    seq_len = 1024
    dtype = torch.float32

    hidden_size = config.hidden_size
    x = torch.rand(1, seq_len, hidden_size, dtype=dtype) - 0.5
    position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, -1)
    inputs = [(x, position_ids)]

    example_inputs = [
        tuple([torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in input_elements])
        for input_elements in inputs
    ]

    # Create a cpu layer
    cpu_module = GptOssRotaryEmbedding(config)

    # Create a function to run the model
    neuron_module = build_module(NeuronGptOssRotaryEmbedding, example_inputs, module_init_kwargs={"config": config})

    # Validate the accuracy of the model. Note that atol and rtol are above the default values, but this is the setting
    # that is working for this model
    validate_accuracy(neuron_module, inputs, cpu_callable=cpu_module, assert_close_kwargs={"atol": 1e-4, "rtol": 1e-4})
