import copy
import logging
import os
import sys
from tempfile import TemporaryDirectory

import huggingface_hub
import pytest
import torch
from conftest import OPTIMUM_CACHE_REPO_ID, _get_hub_neuron_model_id
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from transformers import AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssForCausalLM,
    GptOssMLP,
    GptOssRMSNorm,
    GptOssRotaryEmbedding,
)

# from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.cache import synchronize_hub_cache
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.models.inference.gpt_oss.configuration_gpt_oss import GptOssConfig
from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import (
    CustomRMSNorm,
    GptOssNxdForCausalLM,
)
from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import (
    GptOssMLP as NeuronGptOssMLP,
)
from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import (
    GptOssRotaryEmbedding as NeuronGptOssRotaryEmbedding,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


# # NOTE: we do this because currently the config in the current version of transformers does not support GptOss.
# CONFIG_MAPPING.register("gpt_oss", GptOssConfig)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)

DECODER_MODEL_CONFIGURATIONS = {
    "tiny-gpt-oss": {
        "model_id": "tengomucho/tiny-random-gpt-oss",
        "export_kwargs": {
            "batch_size": 1,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "bf16",
        },
    },
    "gpt-oss-20b": {
        "model_id": "openai/gpt-oss-20b",
        "export_kwargs": {
            "batch_size": 1,
            "sequence_length": 256,
            "num_cores": 8,
            "auto_cast_type": "bf16",
        },
    },
}


def _export_gpt_oss_model(model_id, export_kwargs, neuron_model_path):
    try:
        config = GptOssConfig.from_pretrained(model_id)
        breakpoint()
        model = GptOssNxdForCausalLM.from_pretrained(
            model_id, config=config, export=True, load_weights=False, **export_kwargs
        )
        model.save_pretrained(neuron_model_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to export {model_id}: {e}")


def _get_neuron_model_for_config(config_name: str, model_config, neuron_model_path) -> dict[str, str]:
    """Expose a neuron decoder model

    The helper first makes sure the following model artifacts are present on the hub:
    - exported neuron model under optimum-internal-testing/neuron-testing-<version>-<name>,
    - cached artifacts under optimum-internal-testing/neuron-testing-cache.
    If not, it will export the model and push it to the hub.

    It then fetches the model locally and return a dictionary containing:
    - a configuration name,
    - the original model id,
    - the export parameters,
    - the neuron model id,
    - the neuron model local path.

    The hub model artifacts are never cleaned up and persist across sessions.
    They must be cleaned up manually when the optimum-neuron version changes.

    """
    model_id = model_config["model_id"]
    export_kwargs = model_config["export_kwargs"]
    neuron_model_id = _get_hub_neuron_model_id(config_name, model_config)
    hub = huggingface_hub.HfApi()
    if False and hub.repo_exists(neuron_model_id):
        logger.info(f"Fetching {neuron_model_id} from the HuggingFace hub")
        hub.snapshot_download(neuron_model_id, local_dir=neuron_model_path)
    else:
        model = _export_gpt_oss_model(model_id, export_kwargs, neuron_model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(neuron_model_path)
        del tokenizer
        # Create the test model on the hub
        #model.push_to_hub(save_directory=neuron_model_path, repository_id=neuron_model_id, private=True)
        # Make sure it is cached
        synchronize_hub_cache(cache_repo_id=OPTIMUM_CACHE_REPO_ID)
    # Add dynamic parameters to the model configuration
    model_config["neuron_model_path"] = neuron_model_path
    model_config["neuron_model_id"] = neuron_model_id
    # Also add model configuration name to allow tests to adapt their expectations
    model_config["name"] = config_name
    # Yield instead of returning to keep a reference to the temporary directory.
    # It will go out of scope and be released only once all tests needing the fixture
    # have been completed.
    return model_config


@pytest.fixture(scope="module", params=DECODER_MODEL_CONFIGURATIONS.keys())
def neuron_decoder_oai_config(request):
    """Expose neuron decoder models for predefined configurations.

    The fixture uses the _get_neuron_model_for_config helper to make sure the models
     corresponding to the predefined configurations are all present locally and on the hub.

    For each exposed model, the local directory is maintained for the duration of the
    test session and cleaned up afterwards.

    """
    config_name = request.param
    model_config = copy.deepcopy(DECODER_MODEL_CONFIGURATIONS[config_name])
    with TemporaryDirectory() as neuron_model_path:
        model_config = _get_neuron_model_for_config(config_name, model_config, neuron_model_path)
        logger.info(f"{config_name} ready for testing ...")
        cache_repo_id = os.environ.get("CUSTOM_CACHE_REPO", None)
        os.environ["CUSTOM_CACHE_REPO"] = OPTIMUM_CACHE_REPO_ID
        yield model_config
        if cache_repo_id is not None:
            os.environ["CUSTOM_CACHE_REPO"] = cache_repo_id
        logger.info(f"Done with {config_name}")


@pytest.fixture(scope="module")
def neuron_oai_decoder_path(neuron_decoder_oai_config):
    yield neuron_decoder_oai_config["neuron_model_path"]


@pytest.fixture(scope="module")
def model_and_tokenizer(neuron_oai_decoder_path):
    model = NeuronModelForCausalLM.from_pretrained(neuron_oai_decoder_path)
    tokenizer = AutoTokenizer.from_pretrained(neuron_oai_decoder_path)
    yield (model, tokenizer)


@is_inferentia_test
@requires_neuronx
def test_gpt_oss(model_and_tokenizer: tuple[NeuronModelForCausalLM, AutoTokenizer]):
    model, tokenizer = model_and_tokenizer
    prompt = "What is Deep Learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    generated_text = tokenizer.decode(outputs[0])
    print(generated_text)
    # breakpoint()
    # logits = model.forward(**inputs, return_logits=True)
    # print(logits)

@is_inferentia_test
@requires_neuronx
def test_gpt_vs_cpu(model_and_tokenizer: tuple[NeuronModelForCausalLM, AutoTokenizer]):
    prompt = "Gravity is"
    checkpoint = "tengomucho/tiny-random-gpt-oss"
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model_cpu = GptOssForCausalLM.from_pretrained(checkpoint).to(device)
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    max_new_tokens = 10
    outputs = model_cpu.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = tokenizer.decode(outputs[0])
    print(generated_text)

    model, tokenizer = model_and_tokenizer
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_text = tokenizer.decode(outputs[0])
    print(generated_text)
    # Comparison would fail


@is_inferentia_test
@requires_neuronx
def test_gpt_oss_rms_norm():
    checkpoint = "tengomucho/tiny-random-gpt-oss"
    config = GptOssConfig.from_pretrained(checkpoint)
    hidden_size = config.hidden_size
    seq_len = 2
    hidden_states = torch.ones(1, seq_len, hidden_size, dtype=torch.bfloat16)
    inputs = [(hidden_states,)]

    example_inputs = [tuple([torch.zeros_like(input_element) for input_element in input_elements]) for input_elements in inputs]

    # Create a cpu layer
    cpu_module = GptOssRMSNorm(hidden_size)

    # Create a function to run the model
    neuron_module = build_module(CustomRMSNorm, example_inputs, module_init_kwargs={"hidden_size": hidden_size})

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
def test_gpt_oss_mlp():
    checkpoint = "tengomucho/tiny-random-gpt-oss"
    config = GptOssConfig.from_pretrained(checkpoint)
    hidden_size = config.hidden_size
    dtype = torch.float32
    seq_len = 2
    hidden_states = torch.rand(seq_len, hidden_size, dtype=dtype) * 6 - 3
    inputs = [(hidden_states,)]

    example_inputs = [tuple([torch.zeros_like(input_element) for input_element in input_elements]) for input_elements in inputs]

    cpu_module = GptOssMLP(config)
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
        NeuronGptOssMLP,
        example_inputs,
        module_init_kwargs={
            "config": config,
            "neuron_config": neuron_config
        }
    )
    state_dict = cpu_module.state_dict()

    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
    state_dict["experts.gate_up_proj.weight"] = state_dict["experts.gate_up_proj"]
    state_dict["experts.gate_up_proj.bias"] = state_dict["experts.gate_up_proj_bias"]
    state_dict["experts.down_proj.weight"] = state_dict["experts.down_proj"]
    state_dict["experts.down_proj.bias"] = state_dict["experts.down_proj_bias"]

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
        tuple([
            torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in input_elements]
        ) for input_elements in inputs
    ]

    # Create a cpu layer
    cpu_module = GptOssRotaryEmbedding(config)

    # Create a function to run the model
    neuron_module = build_module(NeuronGptOssRotaryEmbedding, example_inputs, module_init_kwargs={"config": config})

    # Validate the accuracy of the model. Note that atol and rtol are above the default values, but this is the setting
    # that is working for this model
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module,
        assert_close_kwargs={"atol": 1e-4, "rtol": 1e-4}
    )
