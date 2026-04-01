# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
"""Hub cache public API — thin re-exports from bucket_cache."""

import torch

from ...utils import logging
from ..utils.argument_utils import DTYPE_MAPPER
from .bucket_cache import hub_neuronx_cache, lookup_cache  # noqa: F401


# Re-export lookup_cache under the old name for callers that use it
get_hub_cached_entries = lookup_cache

logger = logging.get_logger()


def synchronize_hub_cache(cache_path=None, cache_repo_id=None, non_blocking=False):
    """No-op — immediate sync in hub_neuronx_cache context replaces explicit sync."""
    logger.info("synchronize_hub_cache is a no-op with bucket-based cache (sync happens in context).")


def select_hub_cached_entries(
    model_id: str,
    task: str | None = None,
    cache_repo_id: str | None = None,
    instance_type: str | None = None,
    batch_size: int | None = None,
    sequence_length: int | None = None,
    tensor_parallel_size: int | None = None,
    torch_dtype: str | torch.dtype | None = None,
):
    """Filter cached entries by hardware specs."""
    entries = lookup_cache(model_id=model_id, task=task)
    selected = []
    for entry in entries:
        if instance_type is not None and entry.get("target") != instance_type:
            continue
        if batch_size is not None and entry.get("batch_size") != batch_size:
            continue
        if sequence_length is not None and entry.get("sequence_length") != sequence_length:
            continue
        if tensor_parallel_size is not None and entry.get("tp_degree") != tensor_parallel_size:
            continue
        if torch_dtype is not None:
            target_value = DTYPE_MAPPER.pt(torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
            entry_value = DTYPE_MAPPER.pt(entry.get("torch_dtype", entry.get("dtype")))
            if target_value != entry_value:
                continue
        selected.append(entry)
    return selected
