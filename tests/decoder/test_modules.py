import torch
from nxd_testing import build_module, validate_accuracy
from transformers import AutoConfig, set_seed
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm, LlamaRotaryEmbedding

from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.models.inference.backend.modules.rms_norm import NeuronRMSNorm
from optimum.neuron.models.inference.llama.modeling_llama import Llama3RotaryEmbedding as NeuronLlama3RotaryEmbedding
from optimum.neuron.models.inference.llama.modeling_llama import NeuronLlamaMLP
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
def test_rms_norm():
    set_seed(42)
    config_id = "unsloth/Llama-3.2-1B-Instruct"
    config = AutoConfig.from_pretrained(config_id)
    hidden_size = config.hidden_size
    seq_len = 4096
    # Initialize hidden states to random values between -3 and 3
    hidden_states = torch.rand(seq_len, hidden_size, dtype=torch.bfloat16) * 6 - 3
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
    set_seed(42)
    config_id = "unsloth/Llama-3.2-1B-Instruct"
    config = AutoConfig.from_pretrained(config_id)
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

    # Create a function to run the model
    neuron_config = NxDNeuronConfig(
        batch_size=1,
        sequence_length=seq_len,
    )
    neuron_module = build_module(
        NeuronLlamaMLP, example_inputs, module_init_kwargs={"config": config, "neuron_config": neuron_config}
    )
    state_dict = cpu_module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.bfloat16)

    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")

    weights = [state_dict]
    neuron_module.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model. Note that atol and rtol are relaxed considering calculations in the neuron
    # model are in bfloat16 and errors accumulate.
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module,
        assert_close_kwargs={"atol": torch.finfo(torch.bfloat16).resolution, "rtol": 1e-1},
    )


@is_inferentia_test
@requires_neuronx
def test_llama_rotary_embedding():
    set_seed(42)
    config_id = "unsloth/Llama-3.2-1B-Instruct"
    config = AutoConfig.from_pretrained(config_id)
    hidden_size = config.hidden_size
    seq_len = 4096
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
    cpu_module = LlamaRotaryEmbedding(config)

    def cpu_module_wrapper(x, position_ids):
        cos, sin = cpu_module(x, position_ids)
        return cos, sin

    # Create a neuron equivalent of the cpu module
    head_dim = config.hidden_size // config.num_attention_heads
    neuron_module = build_module(
        NeuronLlama3RotaryEmbedding,
        example_inputs,
        module_init_kwargs={
            "dim": head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "factor": config.rope_scaling["factor"],
            "low_freq_factor": config.rope_scaling["low_freq_factor"],
            "high_freq_factor": config.rope_scaling["high_freq_factor"],
            "original_max_position_embeddings": config.rope_scaling["original_max_position_embeddings"],
        },
    )

    # Validate the accuracy of the model. Note that atol and rtol are relaxed considering calculations in the neuron
    # model are in bfloat16 and errors accumulate.
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module_wrapper,
        assert_close_kwargs={"atol": torch.finfo(torch.bfloat16).resolution, "rtol": 1e-1},
    )
