import os
import shutil
import time
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils import get_hub_cached_entries


def get_export_kwargs_from_env():
    batch_size = os.environ.get("HF_BATCH_SIZE", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    sequence_length = os.environ.get("HF_SEQUENCE_LENGTH", None)
    if sequence_length is not None:
        sequence_length = int(sequence_length)
    num_cores = os.environ.get("HF_NUM_CORES", None)
    if num_cores is not None:
        num_cores = int(num_cores)
    auto_cast_type = os.environ.get("HF_AUTO_CAST_TYPE", None)
    return {
        "task": "text-generation",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "num_cores": num_cores,
        "auto_cast_type": auto_cast_type,
    }


def is_cached(model_id, neuron_config):
    # Look for cached entries for the specified model
    in_cache = False
    entries = get_hub_cached_entries(model_id, "inference")
    # Look for compatible entries
    for entry in entries:
        compatible = True
        for key, value in neuron_config.items():
            # Only weights can be different
            if key in ["checkpoint_id", "checkpoint_revision"]:
                continue
            if entry[key] != value:
                compatible = False
        if compatible:
            in_cache = True
            break
    return in_cache


def log_cache_size():
    path = HF_HUB_CACHE
    if os.path.exists(path):
        usage = shutil.disk_usage(path)
        gb = 2**30
        logger.info(f"Cache disk [{path}]: total = {usage.total/gb:.2f} G, free = {usage.free/gb:.2f} G")
    else:
        raise ValueError(f"The cache directory ({path}) does not exist.")


def fetch_model(
    model_id: str,
    revision: Optional[str] = None,
) -> str:
    """Fetch a neuron model.

    Args:
        model_id (`str`):
            The *model_id* of a model on the HuggingFace hub or the path to a local model.
        revision (`Optional[str]`, defaults to `None`):
            The revision of the model on the HuggingFace hub.

    Returns:
        A string corresponding to the model_id or path.
    """
    if not os.path.isdir("/sys/class/neuron_device/"):
        raise SystemError("No neuron cores detected on the host.")
    if os.path.isdir(model_id) and revision is not None:
        logger.warning("Revision {} ignored for local model at {}".format(revision, model_id))
        revision = None
    # Download the model from the Hub (HUGGING_FACE_HUB_TOKEN must be set for a private or gated model)
    # Note that the model may already be present in the cache.
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    neuron_config = getattr(config, "neuron", None)
    if neuron_config is not None:
        if os.path.isdir(model_id):
            return model_id
        # Prefetch the neuron model from the Hub
        logger.info(f"Fetching revision [{revision}] for neuron model {model_id} under {HF_HUB_CACHE}")
        log_cache_size()
        return snapshot_download(model_id, revision=revision)
    # Model needs to be exported: look for compatible cached entries on the hub
    export_kwargs = get_export_kwargs_from_env()
    export_config = NeuronModelForCausalLM.get_export_config(model_id, config, revision=revision, **export_kwargs)
    neuron_config = export_config.neuron
    if not is_cached(model_id, neuron_config):
        error_msg = (
            f"No cached version found for {model_id} with {neuron_config}."
            "You can start a discussion to request it on https://huggingface.co/aws-neuron/optimum-neuron-cache."
        )
        raise ValueError(error_msg)
    logger.warning(f"{model_id} is not a neuron model: it will be exported using cached artifacts.")
    # Prefetch weights, tokenizer and generation config so that they are in cache
    log_cache_size()
    start = time.time()
    AutoModelForCausalLM.from_pretrained(model_id, revision=revision)
    mid = time.time()
    logger.info(f"Model weights fetched in {mid - start:.2f} s.")
    AutoTokenizer.from_pretrained(model_id, revision=revision)
    end = time.time()
    logger.info(f"Tokenizer fetched in {end - mid:.2f} s.")
    try:
        GenerationConfig.from_pretrained(model_id, revision=revision)
    except Exception:
        logger.warning(f"No default generation config found for {model_id}.")
    log_cache_size()
    return model_id
