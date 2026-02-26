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

from optimum.neuron import NeuronModelForCausalLM, NeuronModelForEmbedding
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

GENERATE_LLM_MODEL_IDS = {
    "llama": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2": "Qwen/Qwen2.5-0.5B",
    "gemma3": "unsloth/gemma-3-270m-it",
    "granite": "ibm-granite/granite-3.1-2b-instruct",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "qwen3": "Qwen/Qwen3-0.6B",
    "smollm3": "HuggingFaceTB/SmolLM3-3B",
}

EMBED_LLM_MODEL_IDS = {
    "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
}


GENERATE_LLM_MODEL_CONFIGURATIONS = {}
for model_name, model_id in GENERATE_LLM_MODEL_IDS.items():
    for batch_size, sequence_length in [(4, 4096), (1, 8192)]:
        GENERATE_LLM_MODEL_CONFIGURATIONS[f"{model_name}-{batch_size}x{sequence_length}"] = {
            "model_id": model_id,
            "task": "text-generation",
            "export_kwargs": {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tensor_parallel_size": cores_per_device(),
            },
        }

EMBED_LLM_MODEL_CONFIGURATIONS = {}
for model_name, model_id in EMBED_LLM_MODEL_IDS.items():
    for batch_size, sequence_length in [(4, 8192), (6, 8192)]:
        EMBED_LLM_MODEL_CONFIGURATIONS[f"{model_name}-{batch_size}x{sequence_length}"] = {
            "model_id": model_id,
            "task": "feature-extraction",
            "export_kwargs": {
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "tensor_parallel_size": cores_per_device(),
            },
        }


LLM_MODEL_CONFIGURATIONS = GENERATE_LLM_MODEL_CONFIGURATIONS | EMBED_LLM_MODEL_CONFIGURATIONS


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


def _export_model(model_id, task, export_kwargs, neuron_model_path):
    if task == "text-generation":
        auto_class = NeuronModelForCausalLM
    elif task == "feature-extraction":
        auto_class = NeuronModelForEmbedding
    else:
        raise ValueError(f"Unsupported task: {task}")
    try:
        neuron_config = auto_class.get_neuron_config(model_id, **export_kwargs)
        model = auto_class.export(model_id, neuron_config=neuron_config, load_weights=False)
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
    task = model_config["task"]
    export_kwargs = model_config["export_kwargs"]
    neuron_model_id = _get_hub_neuron_model_id(config_name, model_config)
    hub = huggingface_hub.HfApi()
    if hub.repo_exists(neuron_model_id):
        logger.info(f"Fetching {neuron_model_id} from the HuggingFace hub")
        hub.snapshot_download(neuron_model_id, local_dir=neuron_model_path)
    else:
        model = _export_model(model_id, task, export_kwargs, neuron_model_path)
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


@pytest.fixture(scope="session", params=GENERATE_LLM_MODEL_CONFIGURATIONS.keys())
def any_generate_model(request):
    """Expose neuron llm generation models for predefined configurations.

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


def _run_exports(configs):
    """Export the given (name, config) pairs with a rich live display if available."""
    total = len(configs)

    try:
        from collections import deque

        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
        from rich.table import Table

        _HAS_RICH = True
    except ImportError:
        _HAS_RICH = False

    if not _HAS_RICH:
        for config_name, model_config in configs:
            with TemporaryDirectory() as neuron_model_path:
                _get_neuron_model_for_config(config_name, model_config, neuron_model_path)
    else:
        LOG_LINES = 10
        log_buffer: deque[str] = deque(maxlen=LOG_LINES)

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        task_id = progress.add_task("Starting...", total=total)

        def make_display():
            grid = Table.grid(padding=(0, 0))
            log_text = "\n".join(log_buffer) if log_buffer else "Waiting for logs..."
            grid.add_row(Panel(log_text, title="Export Logs", height=LOG_LINES + 2))
            grid.add_row(progress)
            return grid

        # Capture log records into the buffer (replaces the default stdout handler)
        class _BufferHandler(logging.Handler):
            def emit(self, record):
                log_buffer.append(self.format(record))

        buf_handler = _BufferHandler()
        buf_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        logging.root.handlers.clear()
        logging.root.addHandler(buf_handler)
        logging.root.setLevel(logging.INFO)

        # Override get_renderable so auto-refresh builds a fresh display each tick
        live = Live(refresh_per_second=4, console=Console(stderr=True))
        live.get_renderable = make_display

        with live:
            for i, (config_name, model_config) in enumerate(configs):
                model_id = model_config["model_id"]
                progress.update(task_id, description=f"[{i + 1}/{total}] {config_name} ({model_id})")
                with TemporaryDirectory() as neuron_model_path:
                    _get_neuron_model_for_config(config_name, model_config, neuron_model_path)
                progress.advance(task_id)
            progress.update(task_id, description="[green]All models exported")
CHUNKED_PREFILL_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
# sequence_length and chunk_size kept small so compilation finishes quickly.
# chunk_size must divide sequence_length.
CHUNKED_PREFILL_SEQUENCE_LENGTH = 4096
CHUNKED_PREFILL_CHUNK_SIZE = 512
CHUNKED_PREFILL_BATCH_SIZE = 2


@pytest.fixture(scope="session")
def chunked_prefill_llm_config():
    """Export and cache a standard + chunked-prefill Llama-3.2-1B model pair.

    Both models share the same base config except for prefill_chunk_size.
    on_device_sampling is disabled on both so the vLLM NeuronSampler is used,
    making the two models directly comparable.

    Yields a dict with keys:
        model_id, std_neuron_model_path, chunk_neuron_model_path,
        chunk_size, sequence_length, batch_size.
    """
    model_id = CHUNKED_PREFILL_MODEL_ID
    tp_degree = cores_per_device()
    std_neuron_model_id = f"{_get_hub_neuron_model_prefix()}-cp-std"
    chunk_neuron_model_id = f"{_get_hub_neuron_model_prefix()}-cp-chunk"

    common_kwargs = {
        "checkpoint_id": model_id,
        "batch_size": CHUNKED_PREFILL_BATCH_SIZE,
        "sequence_length": CHUNKED_PREFILL_SEQUENCE_LENGTH,
        "tp_degree": tp_degree,
        "torch_dtype": "bfloat16",
        "on_device_sampling": False,
        "fused_qkv": True,
        "target": current_instance_type(),
    }

    hub = huggingface_hub.HfApi()
    with TemporaryDirectory() as tmpdir:
        std_path = os.path.join(tmpdir, "std")
        chunk_path = os.path.join(tmpdir, "chunk")

        # --- Standard (context-encoding) model ---
        if hub.repo_exists(std_neuron_model_id):
            logger.info(f"Fetching {std_neuron_model_id} from the HuggingFace hub")
            hub.snapshot_download(std_neuron_model_id, local_dir=std_path)
        else:
            neuron_config = NxDNeuronConfig(**common_kwargs)
            model = NeuronModelForCausalLM.export(model_id, neuron_config=neuron_config, load_weights=False)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model.save_pretrained(std_path)
            tokenizer.save_pretrained(std_path)
            del tokenizer
            model.push_to_hub(save_directory=std_path, repository_id=std_neuron_model_id, private=True)
            synchronize_hub_cache(cache_repo_id=OPTIMUM_CACHE_REPO_ID)

        # --- Chunked-prefill model ---
        if hub.repo_exists(chunk_neuron_model_id):
            logger.info(f"Fetching {chunk_neuron_model_id} from the HuggingFace hub")
            hub.snapshot_download(chunk_neuron_model_id, local_dir=chunk_path)
        else:
            neuron_config = NxDNeuronConfig(**common_kwargs, prefill_chunk_size=CHUNKED_PREFILL_CHUNK_SIZE)
            model = NeuronModelForCausalLM.export(model_id, neuron_config=neuron_config, load_weights=False)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model.save_pretrained(chunk_path)
            tokenizer.save_pretrained(chunk_path)
            del tokenizer
            model.push_to_hub(save_directory=chunk_path, repository_id=chunk_neuron_model_id, private=True)
            synchronize_hub_cache(cache_repo_id=OPTIMUM_CACHE_REPO_ID)

        yield {
            "model_id": model_id,
            "std_neuron_model_path": std_path,
            "chunk_neuron_model_path": chunk_path,
            "chunk_size": CHUNKED_PREFILL_CHUNK_SIZE,
            "sequence_length": CHUNKED_PREFILL_SEQUENCE_LENGTH,
            "batch_size": CHUNKED_PREFILL_BATCH_SIZE,
        }


if __name__ == "__main__":
    import argparse
    import fnmatch

    parser = argparse.ArgumentParser(description="Export Neuron test models")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available configuration names and exit",
    )
    parser.add_argument(
        "pattern",
        nargs="?",
        default="*",
        help="Glob pattern to filter configurations (e.g. 'gemma*', '*-1x8192')",
    )
    args = parser.parse_args()

    all_configs = list(LLM_MODEL_CONFIGURATIONS.items())
    configs = [(name, cfg) for name, cfg in all_configs if fnmatch.fnmatch(name, args.pattern)]

    if args.list:
        for name, cfg in configs:
            print(f"{name:30s}  {cfg['model_id']}")
        sys.exit(0)

    if not configs:
        print(f"No configurations match pattern '{args.pattern}'")
        print(f"Available: {', '.join(name for name, _ in all_configs)}")
        sys.exit(1)

    _run_exports(configs)
