import os
import shutil
import time
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from loguru import logger
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils import ModelCacheEntry, get_hub_cached_entries


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
        Local folder path (string) of the model.
    """
    if not os.path.isdir("/sys/class/neuron_device/"):
        raise SystemError("No neuron cores detected on the host.")
    if os.path.isdir(model_id):
        if revision is not None:
            logger.warning("Revision {} ignored for local model at {}".format(revision, model_id))
        return model_id
    # Download the model from the Hub (HUGGING_FACE_HUB_TOKEN must be set for a private or gated model)
    # Note that the model may already be present in the cache.
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    neuron_config = getattr(config, "neuron", None)
    log_cache_size()
    if neuron_config is not None:
        logger.info(f"Fetching revision [{revision}] for neuron model {model_id} under {HF_HUB_CACHE}")
        return snapshot_download(model_id, revision=revision)
    # Not a neuron model: evaluate the export config and check if it has been exported locally
    export_kwargs = get_export_kwargs_from_env()
    export_config = NeuronModelForCausalLM.get_export_config(model_id, config, revision=revision, **export_kwargs)
    entry = ModelCacheEntry(model_id, export_config)
    export_path = f"{HF_HUB_CACHE}/{entry.hash}"
    if os.path.exists(export_path):
        # The model has already been exported for that configuration
        logger.info(f"Neuron model for {model_id} with {export_config.neuron} found under {export_path}.")
        return export_path
    # Look for compatible cached entries on the hub
    neuron_config = export_config.neuron
    if not is_cached(model_id, neuron_config):
        error_msg = (
            f"No cached version found for {model_id} with {neuron_config}."
            "You can start a discussion to request it on https://huggingface.co/aws-neuron/optimum-neuron-cache."
        )
        raise ValueError(error_msg)
    # Export the model
    logger.warning(f"{model_id} is not a neuron model: it will be exported using cached artifacts.")
    start = time.time()
    logger.info(f"Exporting model to neuron with config {neuron_config}.")
    log_cache_size()
    start = time.time()
    model = NeuronModelForCausalLM.from_pretrained(model_id, export=True, **export_kwargs)
    end = time.time()
    logger.info(f"Model successfully exported in {end - start:.2f} s.")
    logger.info(f"Saving exported model to local storage under {export_path}.")
    log_cache_size()
    model.save_pretrained(export_path)
    logger.info(f"Saving model tokenizer under {export_path}.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.save_pretrained(export_path)
    try:
        config = GenerationConfig.from_pretrained(model_id, revision=revision)
        config.save_pretrained(export_path)
        logger.info(f"Saved model default generation config under {export_path}.")
    except Exception:
        logger.warning(f"No default generation config found for {model_id}.")
    logger.info(f"Model successfully exported in {end - start:.2f} s under {export_path}.")
    return export_path
