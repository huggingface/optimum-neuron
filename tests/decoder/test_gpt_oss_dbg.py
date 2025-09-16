import copy
import logging
import os
import sys
from tempfile import TemporaryDirectory

import huggingface_hub
import pytest
import torch
from conftest import OPTIMUM_CACHE_REPO_ID, _get_hub_neuron_model_id
from nxd_testing import build_module, validate_accuracy
from transformers import AutoConfig, AutoTokenizer, set_seed
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssModel,
)

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.cache import synchronize_hub_cache
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import (
    NeuronGptOssDecoderLayer,
    NeuronGptOssModel,
    NeuronRMSNorm,
    ParallelEmbedding,
    convert_gptoss_to_neuron_state_dict,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


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
            "tensor_parallel_size": 2,
            "auto_cast_type": "bf16",
        },
    },
    # "gpt-oss-20b": {
    #     "model_id": "openai/gpt-oss-20b",
    #     "export_kwargs": {
    #         "batch_size": 1,
    #         "sequence_length": 256,
    #         "tensor_parallel_size": 8,
    #         "auto_cast_type": "bf16",
    #     },
    # },
}


def _export_gpt_oss_model(model_id, export_kwargs, neuron_model_path):
    try:
        neuron_config = NeuronModelForCausalLM.get_neuron_config(model_id, **export_kwargs)
        model = NeuronModelForCausalLM.export(model_id, neuron_config=neuron_config, export=True, load_weights=False)
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
        model = _export_gpt_oss_model(model_id, export_kwargs, neuron_model_path)  # noqa: F841
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(neuron_model_path)
        del tokenizer
        # Create the test model on the hub
        # model.push_to_hub(save_directory=neuron_model_path, repository_id=neuron_model_id, private=True)
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


def _set_weights(module):
    """Set the weights of the module to random values"""
    state_dict = module.state_dict()
    for key in list(state_dict.keys()):
        state_dict[key] = torch.nn.Parameter(torch.rand_like(state_dict[key]) * 0.1)
    module.load_state_dict(state_dict)


def _masked_attention_mask(attention_mask, config):
    seq_len = attention_mask.shape[-1]
    sliding_window = config.sliding_window
    sliding_window_overlay = (
        torch.ones(seq_len, seq_len, device=attention_mask.device).to(torch.bool).triu(diagonal=1 - sliding_window)
    )
    sliding_window_overlay = sliding_window_overlay[None, None, :, :].expand(
        attention_mask.shape[0], 1, seq_len, seq_len
    )
    sliding_window_mask = torch.logical_and(attention_mask, sliding_window_overlay)
    return sliding_window_mask


def _full_attention_mask(attention_mask, _config):
    return attention_mask


def _attention_mask_functions(attention_mask, index, config):
    if config.layer_types[index] == "sliding_attention":
        return _masked_attention_mask(attention_mask, config)
    elif config.layer_types[index] == "full_attention":
        return _full_attention_mask(attention_mask, config)
    else:
        raise ValueError(f"Unsupported layer type: {config.layer_types[index]}")


class GptOssModelWrapper(GptOssModel):
    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        # seq_ids and sampling_params are ignored
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return outputs.last_hidden_state


class NeuronDecoderLayersWrapper_(torch.nn.Module):
    def __init__(self, config, neuron_config):
        super().__init__()
        self.config = config
        self.neuron_config = neuron_config

        # self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            # dtype=neuron_config.torch_dtype,
            dtype=torch.float32,
            shard_across_embedding=True,
            pad=True,
        )

        self.layers = torch.nn.ModuleList(
            [NeuronGptOssDecoderLayer(config, neuron_config) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        # seq_ids and sampling_params are ignored
        cos_cache = None
        sin_cache = None

        hidden_states = self.embed_tokens(input_ids)
        for layer_idx in range(self.config.num_hidden_layers):
            layer = self.layers[layer_idx]
            attention_mask_layer = _attention_mask_functions(attention_mask, layer_idx, self.config)
            hidden_states, _next_decoder_cache, cos_cache, sin_cache = layer(
                hidden_states,
                attention_mask_layer,
                position_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class NeuronGptOssModelWrapper(NeuronGptOssModel):
    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        return self.get_model_output(input_ids, attention_mask, position_ids)[0]


@is_inferentia_test
@requires_neuronx
def test_gpt_oss_model():
    set_seed(42)
    config_id = "tengomucho/tiny-random-gpt-oss"
    config = AutoConfig.from_pretrained(config_id)
    seq_len = 128
    # Initialize hidden states to random values between -3 and 3
    prompt = "Gravity is"
    tokenizer = AutoTokenizer.from_pretrained(config_id)
    tokenizer.pad_token = "!"
    inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=seq_len)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    seq_len = input_ids.shape[-1]
    position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)
    seq_ids = torch.arange(input_ids.shape[0])
    sampling_params = torch.tensor([1.0, 1.0, 1.0])  # This is ignored in the model

    if getattr(config, "sliding_window", False):
        # If model config supports sliding window, test it, and set it to a value that can be tested with the given
        # sequence length.
        config.sliding_window = min(seq_len // 4, config.sliding_window)

    inputs = [(input_ids, attention_mask, position_ids, seq_ids, sampling_params)]
    example_inputs = [
        tuple([torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in inputs[0]])
    ]

    config._attn_implementation = "eager"  # Force eager attention in cpu
    cpu_module = GptOssModelWrapper(config)
    _set_weights(cpu_module)

    # Create a function to run the model
    neuron_config = NxDNeuronConfig(
        batch_size=1,
        sequence_length=seq_len,
    )
    state_dict = cpu_module.state_dict()
    state_dict = convert_gptoss_to_neuron_state_dict(state_dict, config, neuron_config)

    with TemporaryDirectory() as tmpdir:
        # There are many quirks in the neuron attention implementation, so we will just save the state dict and load it
        # again to build the module.
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(state_dict, checkpoint_path)

        neuron_module = build_module(
            NeuronGptOssModelWrapper,
            example_inputs,
            module_init_kwargs={"config": config, "neuron_config": neuron_config},
            checkpoint_path=checkpoint_path,
        )

    # Validate the accuracy of the model
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module,
        assert_close_kwargs={"atol": torch.finfo(torch.bfloat16).resolution, "rtol": 1e-1},
    )
