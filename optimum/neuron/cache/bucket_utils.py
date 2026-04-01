# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for HF Storage Bucket-based NEFF cache."""

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

from ...utils import logging
from ..utils.misc import is_main_worker


logger = logging.get_logger()

CACHE_BUCKET_FILENAME = "optimum_neuron_cache_bucket"
HF_HOME_CACHE_BUCKET_FILE = f"{HF_HOME}/{CACHE_BUCKET_FILENAME}"

DEFAULT_CACHE_BUCKET = "aws-neuron/optimum-neuron-neff-cache"


def get_cache_bucket() -> str | None:
    """Resolve the cache bucket ID.

    Priority: NEURON_CACHE_BUCKET env var > locally saved > default.
    """
    bucket = os.environ.get("NEURON_CACHE_BUCKET")
    if bucket:
        return bucket
    bucket = load_cache_bucket_from_hf_home()
    if bucket:
        return bucket
    return DEFAULT_CACHE_BUCKET


def load_cache_bucket_from_hf_home(
    hf_home_cache_bucket_file: str | Path | None = None,
) -> str | None:
    if hf_home_cache_bucket_file is None:
        hf_home_cache_bucket_file = HF_HOME_CACHE_BUCKET_FILE
    path = Path(hf_home_cache_bucket_file)
    if path.exists():
        with open(path, "r") as fp:
            bucket_id = fp.read().strip()
            return bucket_id if bucket_id else None
    return None


def set_cache_bucket_in_hf_home(
    bucket_id: str,
    hf_home: str = HF_HOME,
):
    """Persist the cache bucket choice locally."""
    hf_home_cache_bucket_file = f"{hf_home}/{CACHE_BUCKET_FILENAME}"

    existing = load_cache_bucket_from_hf_home(hf_home_cache_bucket_file)
    if is_main_worker() and existing is not None:
        logger.warning(
            f"A custom cache bucket was already registered: {existing}. It will be overwritten to {bucket_id}."
        )

    with open(hf_home_cache_bucket_file, "w") as fp:
        fp.write(bucket_id)


def config_hash(neuron_config: dict) -> str:
    """8-char SHA256 of the JSON-serialized sorted neuron_config."""
    serialized = json.dumps(neuron_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]


def bucket_flat_prefix(compiler_version: str) -> str:
    """Bucket prefix for the flat NEFF area (hf-mount compatible).

    Maps to: neuronxcc-{compiler_version}/
    This matches the local cache layout exactly.
    """
    return f"neuronxcc-{compiler_version}"


def bucket_model_prefix(compiler_version: str, model_id: str) -> str:
    """Bucket prefix for the per-model area.

    Maps to: neuronxcc-{compiler_version}/{org}/{model}/
    Uses natural `/` separators for org-level operations (e.g. rebuild cache for an org).
    Local model names without org use the model name directly.
    """
    return f"neuronxcc-{compiler_version}/{model_id}"


def local_to_flat_bucket_path(module_dir_name: str, compiler_version: str) -> str:
    """Convert a local MODULE dir name to its flat bucket path.

    Args:
        module_dir_name: e.g. "MODULE_X+Y"
        compiler_version: e.g. "2.28.4405.0+abc123"

    Returns: e.g. "neuronxcc-2.28.4405.0+abc123/MODULE_X+Y"
    """
    return f"{bucket_flat_prefix(compiler_version)}/{module_dir_name}"
