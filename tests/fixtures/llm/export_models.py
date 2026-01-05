import copy
import hashlib
import logging
import os
import sys
from tempfile import TemporaryDirectory

import huggingface_hub
import pytest

from optimum.neuron.utils.import_utils import is_package_available
from optimum.neuron.utils.instance import current_instance_type
from optimum.neuron.utils.system import cores_per_device


if is_package_available("transformers"):
    from transformers import AutoConfig, AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.cache import synchronize_hub_cache
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.version import __sdk_version__ as sdk_version
from optimum.neuron.version import __version__ as version


TEST_ORGANIZATION = "optimum-internal-testing"
TEST_CACHE_REPO_ID = f"{TEST_ORGANIZATION}/neuron-testing-cache"
HF_TOKEN = huggingface_hub.get_token()


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)

TEST_HUB_ORG = os.getenv("TEST_HUB_ORG", "optimum-internal-testing")
OPTIMUM_CACHE_REPO_ID = f"{TEST_HUB_ORG}/neuron-testing-cache"

LLM_MODEL_IDS = {
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "granite": "ibm-granite/granite-3.1-2b-instruct",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
}

LLM_MODEL_CONFIGURATIONS = {}

for model_name, model_id in LLM_MODEL_IDS.items():
    for batch_size, sequence_length in [(4, 4096), (1, 8192)]:
        LLM_MODEL_CONFIGURATIONS[f"{model_name}-{batch_size}x{sequence_length}"] = {
            "model_id": model_id,
            "export_kwargs": {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tensor_parallel_size": cores_per_device(),
            },
        }


def get_neuron_models_hash():
    import subprocess

    res = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    root_dir = res.stdout.split("\n")[0]

    def get_sha(path):
        res = subprocess.run(
            ["git", "ls-tree", "HEAD", f"{root_dir}/{path}"],
            capture_output=True,
            text=True,
        )
        # Output of the command is in the form '040000 tree|blob <SHA>\t<path>\n'
        sha = res.stdout.split("\t")[0].split(" ")[-1]
        return sha.encode()

    # We hash both the neuron models directory and setup file and create a smaller hash out of that
    m = hashlib.sha256()
    m.update(get_sha("pyproject.toml"))
    m.update(get_sha("optimum/neuron/models/inference"))
    return m.hexdigest()[:10]


def _get_hub_neuron_model_prefix():
    return f"{TEST_HUB_ORG}/optimum-neuron-testing-{version}-{sdk_version}-{current_instance_type()}-{get_neuron_models_hash()}"


def _get_hub_neuron_model_id(config_name: str, model_config: dict[str, str]):
    return f"{_get_hub_neuron_model_prefix()}-{config_name}"


def _export_model(model_id, export_kwargs, neuron_model_path):
    try:
        neuron_config = NeuronModelForCausalLM.get_neuron_config(model_id, **export_kwargs)
        model = NeuronModelForCausalLM.export(model_id, neuron_config=neuron_config, load_weights=False)
        model.save_pretrained(neuron_model_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to export {model_id}: {e}")


def _get_neuron_model_for_config(config_name: str, model_config, neuron_model_path) -> dict[str, str]:
    """Expose a neuron llm model

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


@pytest.fixture(scope="session", params=LLM_MODEL_CONFIGURATIONS.keys())
def any_neuron_llm_config(request):
    """Expose neuron llm models for predefined configurations.

    The fixture uses the _get_neuron_model_for_config helper to make sure the models
     corresponding to the predefined configurations are all present locally and on the hub.

    For each exposed model, the local directory is maintained for the duration of the
    test session and cleaned up afterwards.

    """
    config_name = request.param
    model_config = copy.deepcopy(LLM_MODEL_CONFIGURATIONS[config_name])
    with TemporaryDirectory() as neuron_model_path:
        model_config = _get_neuron_model_for_config(config_name, model_config, neuron_model_path)
        cache_repo_id = os.environ.get("CUSTOM_CACHE_REPO", None)
        os.environ["CUSTOM_CACHE_REPO"] = OPTIMUM_CACHE_REPO_ID
        yield model_config
        if cache_repo_id is not None:
            os.environ["CUSTOM_CACHE_REPO"] = cache_repo_id


@pytest.fixture(scope="session")
def neuron_llm_config(request):
    """Expose a base neuron llm model path for testing purposes.

    This fixture is used to test the export of models that do not have a predefined configuration.
    It will create a temporary directory and yield its path.

    If the param is not provided, it will use the first model configuration in the list.
    """
    first_config_name = list(LLM_MODEL_CONFIGURATIONS.keys())[0]
    config_name = getattr(request, "param", first_config_name)
    model_config = copy.deepcopy(LLM_MODEL_CONFIGURATIONS[config_name])
    with TemporaryDirectory() as neuron_model_path:
        neuron_model_config = _get_neuron_model_for_config(config_name, model_config, neuron_model_path)
        cache_repo_id = os.environ.get("CUSTOM_CACHE_REPO", None)
        os.environ["CUSTOM_CACHE_REPO"] = OPTIMUM_CACHE_REPO_ID
        yield neuron_model_config
        if cache_repo_id is not None:
            os.environ["CUSTOM_CACHE_REPO"] = cache_repo_id


@pytest.fixture(scope="session")
def speculation():
    model_id = "unsloth/Llama-3.2-1B-Instruct"
    neuron_model_id = f"{_get_hub_neuron_model_prefix()}-speculation"
    draft_neuron_model_id = f"{_get_hub_neuron_model_prefix()}-speculation-draft"
    tp_degree = cores_per_device()
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
                tp_degree=tp_degree,
                torch_dtype="bf16",
                target=current_instance_type(),
                speculation_length=5,
            )
            model = NeuronModelForCausalLM.export(
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
                tp_degree=tp_degree,
                torch_dtype="bf16",
                target=current_instance_type(),
            )
            model = NeuronModelForCausalLM.export(
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


if __name__ == "__main__":
    for config_name, model_config in LLM_MODEL_CONFIGURATIONS.items():
        with TemporaryDirectory() as neuron_model_path:
            _get_neuron_model_for_config(config_name, model_config, neuron_model_path)
