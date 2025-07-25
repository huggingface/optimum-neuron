import copy
import logging
import os
import sys
from tempfile import TemporaryDirectory

import huggingface_hub
import pytest
from conftest import OPTIMUM_CACHE_REPO_ID, _get_hub_neuron_model_id
from transformers import AutoTokenizer
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.cache import synchronize_hub_cache
from optimum.neuron.models.inference.gpt_oss.configuration_gpt_oss import GptOssConfig
from optimum.neuron.models.inference.gpt_oss.modeling_gpt_oss import GptOssNxdForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


# NOTE: we do this because currently the config in the current version of transformers does not support GptOss.
CONFIG_MAPPING.register("gpt_oss", GptOssConfig)

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
    "gpt-rc-20b": {
        "model_id": "plop-internal/gpt-oss-20b-trfs-0804",
        "export_kwargs": {
            "batch_size": 1,
            "sequence_length": 4096,
            "num_cores": 16,
            "auto_cast_type": "bf16",
        },
    },
}


def _export_gpt_oss_model(model_id, export_kwargs, neuron_model_path):
    try:
        config = GptOssConfig.from_pretrained(model_id)
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
    if hub.repo_exists(neuron_model_id):
        logger.info(f"Fetching {neuron_model_id} from the HuggingFace hub")
        hub.snapshot_download(neuron_model_id, local_dir=neuron_model_path)
    else:
        model = _export_gpt_oss_model(model_id, export_kwargs, neuron_model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(neuron_model_path)
        del tokenizer
        # Create the test model on the hub
        model.push_to_hub(save_directory=neuron_model_path, repository_id=neuron_model_id, private=True)
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
def test_gpt_oss(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "What is Deep Learning?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated_text = tokenizer.decode(outputs[0])
    print(generated_text)
