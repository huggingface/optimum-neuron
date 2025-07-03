import copy
import logging
import os
import sys
from tempfile import TemporaryDirectory

import huggingface_hub
import pytest
from transformers import AutoConfig, AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.cache import synchronize_hub_cache
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.models.inference.llama.modeling_llama import LlamaNxDModelForCausalLM
from optimum.neuron.version import __sdk_version__ as sdk_version
from optimum.neuron.version import __version__ as version


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)

OPTIMUM_CACHE_REPO_ID = "optimum-internal-testing/neuron-testing-cache"

# All model configurations below will be added to the neuron_model_config fixture
DECODER_MODEL_CONFIGURATIONS = {
    "llama": {
        "model_id": "unsloth/Llama-3.2-1B-Instruct",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "fp16",
        },
    },
    "qwen2": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "fp16",
        },
    },
    "granite": {
        "model_id": "ibm-granite/granite-3.1-2b-instruct",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "bf16",
        },
    },
    "phi": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "bf16",
        },
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B",
        "export_kwargs": {
            "batch_size": 4,
            "sequence_length": 4096,
            "num_cores": 2,
            "auto_cast_type": "bf16",
        },
    },
}


def _get_hub_neuron_model_prefix():
    """Get the prefix for the neuron model id on the hub"""
    return f"optimum-internal-testing/neuron-testing-{version}-{sdk_version}"


def _get_hub_neuron_model_id(config_name: str, model_config: dict[str, str]):
    return f"{_get_hub_neuron_model_prefix()}-{config_name}"


def _export_model(model_id, export_kwargs, neuron_model_path):
    try:
        model = NeuronModelForCausalLM.from_pretrained(model_id, export=True, load_weights=False, **export_kwargs)
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
        model = _export_model(model_id, export_kwargs, neuron_model_path)
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


@pytest.fixture(scope="session", params=DECODER_MODEL_CONFIGURATIONS.keys())
def neuron_decoder_config(request):
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
def neuron_decoder_path(neuron_decoder_config):
    yield neuron_decoder_config["neuron_model_path"]


@pytest.fixture(scope="module")
def base_neuron_decoder_config():
    """Expose a base neuron model path for testing purposes.

    This fixture is used to test the export of models that do not have a predefined configuration.
    It will create a temporary directory and yield its path.
    """
    with TemporaryDirectory() as neuron_model_path:
        model_config = {
            "model_id": "Qwen/Qwen2.5-0.5B",
            "export_kwargs": {
                "batch_size": 1,
                "sequence_length": 4096,
                "num_cores": 2,
                "auto_cast_type": "bf16",
            },
        }
        neuron_model_config = _get_neuron_model_for_config("base", model_config, neuron_model_path)
        logger.info("Base neuron model ready for testing ...")
        yield neuron_model_config
        logger.info("Done with base neuron model testing")


@pytest.fixture(scope="module")
def base_neuron_decoder_path(base_neuron_decoder_config):
    yield base_neuron_decoder_config["neuron_model_path"]


@pytest.fixture(scope="module")
def speculation():
    model_id = "unsloth/Llama-3.2-1B-Instruct"
    neuron_model_id = f"{_get_hub_neuron_model_prefix()}-speculation"
    draft_neuron_model_id = f"{_get_hub_neuron_model_prefix()}-speculation-draft"
    with TemporaryDirectory() as speculation_path:
        hub = huggingface_hub.HfApi()
        neuron_model_path = os.path.join(speculation_path, "model")
        if hub.repo_exists(neuron_model_id):
            logger.info(f"Fetching {neuron_model_id} from the HuggingFace hub")
            hub.snapshot_download(neuron_model_id, local_dir=neuron_model_path)
        else:
            neuron_config = NxDNeuronConfig(
                checkpoint_id=model_id,
                batch_size=1,
                sequence_length=4096,
                tp_degree=2,
                torch_dtype="bf16",
                speculation_length=5,
            )
            model = LlamaNxDModelForCausalLM.export(
                model_id,
                config=AutoConfig.from_pretrained(model_id),
                neuron_config=neuron_config,
                load_weights=False,
            )
            model.save_pretrained(neuron_model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            tokenizer.save_pretrained(neuron_model_path)
            del tokenizer
            # Create the speculation model on the hub
            model.push_to_hub(save_directory=neuron_model_path, repository_id=neuron_model_id, private=False)
            # Make sure it is cached
            synchronize_hub_cache(cache_repo_id=OPTIMUM_CACHE_REPO_ID)
        draft_neuron_model_path = os.path.join(speculation_path, "draft-model")
        if hub.repo_exists(draft_neuron_model_id):
            logger.info(f"Fetching {draft_neuron_model_id} from the HuggingFace hub")
            hub.snapshot_download(draft_neuron_model_id, local_dir=draft_neuron_model_path)
        else:
            neuron_config = NxDNeuronConfig(
                checkpoint_id=model_id,
                batch_size=1,
                sequence_length=4096,
                tp_degree=2,
                torch_dtype="bf16",
            )
            model = LlamaNxDModelForCausalLM.export(
                model_id,
                config=AutoConfig.from_pretrained(model_id),
                neuron_config=neuron_config,
                load_weights=False,
            )
            model.save_pretrained(draft_neuron_model_path)
            # Create the draft model on the hub
            model.push_to_hub(
                save_directory=draft_neuron_model_path, repository_id=draft_neuron_model_id, private=False
            )
            # Make sure it is cached
            synchronize_hub_cache(cache_repo_id=OPTIMUM_CACHE_REPO_ID)
        yield neuron_model_path, draft_neuron_model_path
        logger.info(f"Done with speculation models at {speculation_path}")
