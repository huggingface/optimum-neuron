import os
from typing import Optional

from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoConfig


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
    if os.path.isdir(model_id):
        if revision is not None:
            logger.warning("Revision {} ignored for local model at {}".format(revision, model_id))
        model_path = model_id
    else:
        # Download the model from the Hub (HUGGING_FACE_HUB_TOKEN must be set for a private or gated model)
        # Note that the model may already be present in the cache.
        logger.info("Fetching revision {} for {}".format(revision, model_id))
        model_path = snapshot_download(model_id, revision=revision)
    config = AutoConfig.from_pretrained(model_path)
    neuron_config = getattr(config, "neuron", None)
    if neuron_config is None:
        raise ValueError("The target model is not a Neuron model. Please export it to neuron first.")
    return model_path
