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
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from huggingface_hub import HfApi, get_token
from transformers import AutoConfig, PretrainedConfig

from ..version import __version__
from .import_utils import is_neuronx_available
from .patching import patch_everywhere
from .require_utils import requires_torch_neuronx


if is_neuronx_available():
    from libneuronxla.neuron_cc_cache import (
        CacheUrl,
        CompileCache,
        CompileCacheFs,
        CompileCacheS3,
        create_compile_cache,
    )
else:

    class CacheUrl:
        pass

    class CompileCache:
        pass

    class CompileCacheFs:
        pass

    class CompileCacheS3:
        pass

    def create_compile_cache():
        pass


logger = logging.getLogger(__name__)


class CompileCacheHfProxy(CompileCache):
    """A HuggingFace Hub proxy cache implementing the CompileCache API.

    This cache first looks for compilation artifacts into the default cache, then the
    specified Hugging Face cache repository.

    Args:
        repo_id (`str`):
            The id of the Hugging Face cache repository, in the form 'org|user/name'.
        default_cache (`CompileCache`):
            The default neuron compiler cache (can be either a local file or S3 cache).
        endpoint (`Optional[str]`, defaults to None):
            The HuggingFaceHub endpoint: only required for unit tests to switch to the staging Hub.
        token (`Optional[str]`, defaults to None):
            The HuggingFace token to use to fetch/push artifacts. If not specified it will correspond
            to the current user token.
    """

    cache_type = "hf"

    def __init__(
        self, repo_id: str, default_cache: CompileCache, endpoint: Optional[str] = None, token: Optional[str] = None
    ):
        # Initialize the proxy cache as expected by the parent class
        super().__init__(default_cache.cache_url)
        self.cache_path = default_cache.cache_path
        # Initialize specific members
        self.default_cache = default_cache
        self.api = HfApi(endpoint=endpoint, token=token, library_name="optimum-neuron", library_version=__version__)
        # Check if the HF cache id is valid
        try:
            if not self.api.repo_exists(repo_id):
                raise ValueError(f"The {repo_id} repository does not exist or you don't have access to it.")
        except Exception as e:
            raise ValueError(f"Error while accessing the {repo_id} cache repository: {e}")
        self.repo_id = repo_id

    def get_cache_dir(self, model_hash: str, compile_flags_str: str):
        return self.default_cache.get_cache_dir(model_hash, compile_flags_str)

    def clean(self):
        self.default_cache.clean()

    def clear_locks(self):
        # Clear locks in the default cache only, as the Hf already deals with concurrency
        self.default_cache.clear_locks()

    def get_hlos(self, failed_neff_str: str = ""):
        return self.default_cache.get_hlos(failed_neff_str)

    def hlo_acquire_lock(self, h: str):
        # Put a lock in the default cache only, as the Hf already deals with concurrency
        return self.default_cache.hlo_acquire_lock(h)

    def hlo_release_lock(self, h: str):
        # Release lock in the default cache only, as the Hf already deals with concurrency
        return self.default_cache.hlo_release_lock(h)

    def remove(self, path: str):
        # Only remove in the default cache
        return self.default_cache.remove(path)

    def _rel_path(self, path: str):
        # Remove the default cache url from the path
        if path.startswith(self.default_cache.cache_path):
            return path[len(self.default_cache.cache_path) :].lstrip("/")

    def exists(self, path: str):
        # Always prioritize the default cache
        if self.default_cache.exists(path):
            return True
        rel_path = self._rel_path(path)
        exists = self.api.file_exists(self.repo_id, rel_path)
        if not exists:
            logger.warning(
                f"{rel_path} not found in {self.repo_id}: the corresponding graph will be recompiled."
                " This may take up to one hour for large models."
            )
        return exists

    def download_file(self, filename: str, dst_path: str):
        # Always prioritize the default cache for faster retrieval
        if self.default_cache.exists(filename):
            self.default_cache.download_file(filename, dst_path)
        else:
            rel_filename = self._rel_path(filename)
            local_path = self.api.hf_hub_download(self.repo_id, rel_filename)
            os.symlink(local_path, dst_path)
            logger.info(f"Fetched cached {rel_filename} from {self.repo_id}")

    def synchronize(self):
        if isinstance(self.default_cache, CompileCacheS3):
            raise ValueError("Hugging Face hub compiler cache synchronization is not supported for S3.")
        logger.info(f"Synchronizing {self.repo_id} Hub cache with {self.default_cache.cache_path} local cache")
        self.api.upload_folder(
            repo_id=self.repo_id,
            folder_path=self.default_cache.cache_path,
            commit_message="Synchronizing local compiler cache.",
            ignore_patterns="lock",
        )
        logger.info("Synchronization complete.")

    def upload_file(self, cache_path: str, src_path: str):
        # Only upload to the default cache: use synchronize to populate the Hub cache
        self.default_cache.upload_file(cache_path, src_path)

    def upload_string_to_file(self, cache_path: str, data: str):
        # Only upload to the default cache: use synchronize to populate the Hub cache
        self.default_cache.upload_string_to_file(cache_path, data)

    def download_file_to_string(self, filename: str, limit: int = None):
        # Always prioritize the default cache for faster retrieval
        if self.default_cache.exists(filename):
            return self.default_cache.download_file_to_string(filename, limit)
        rel_filename = self._rel_path(filename)
        local_path = self.api.hf_hub_download(self.repo_id, rel_filename)
        with open(local_path, "rb") as f:
            s = f.read().decode(errors="replace")
        logger.info(f"Fetched cached {rel_filename} from {self.repo_id}")
        return s


def get_hub_cache():
    HUB_CACHE = "aws-neuron/optimum-neuron-cache"
    return os.getenv("CUSTOM_CACHE_REPO", HUB_CACHE)


def _create_hub_compile_cache_proxy(
    cache_url: Optional[CacheUrl] = None,
    cache_repo_id: Optional[str] = None,
):
    if cache_url is None:
        cache_url = CacheUrl.get_cache_url()
    if cache_repo_id is None:
        cache_repo_id = get_hub_cache()
    default_cache = CompileCacheS3(cache_url) if cache_url.is_s3() else CompileCacheFs(cache_url)
    # Reevaluate endpoint and token (needed for tests altering the environment)
    endpoint = os.getenv("HF_ENDPOINT")
    token = get_token()
    return CompileCacheHfProxy(cache_repo_id, default_cache, endpoint=endpoint, token=token)


class ModelCacheEntry:
    """A class describing a model cache entry

    Args:
        model_id (`str`):
            The model id, used as a key for the cache entry.
        config (`transformers.PretrainedConfig`):
            The configuration of the model.

    """

    def __init__(self, model_id: str, config: PretrainedConfig):
        self.model_id = model_id
        # Remove keys set to default values
        self.config = config.to_diff_dict()
        excluded_keys = ["_name_or_path", "transformers_version"]
        for key in excluded_keys:
            self.config.pop(key, None)

    def to_json(self) -> str:
        return json.dumps(self.config)

    @property
    def hash(self):
        hash_gen = hashlib.sha512()
        hash_gen.update(self.to_json().encode("utf-8"))
        return str(hash_gen.hexdigest())[:20]


REGISTRY_FOLDER = f"0_REGISTRY/{__version__}"


@requires_torch_neuronx
@contextmanager
def hub_neuronx_cache(entry: Optional[ModelCacheEntry] = None):
    """A context manager to activate the Hugging Face Hub proxy compiler cache.

    Args:
        entry (`Optional[ModelCacheEntry]`, defaults to `None`):
            An optional dataclass containing metadata associated with the model corresponding
            to the cache session. Will create a dedicated entry in the cache registry.
    """

    def hf_create_compile_cache(cache_url):
        try:
            return _create_hub_compile_cache_proxy(cache_url)
        except Exception as e:
            logger.warning(f"Bypassing Hub cache because of the following error: {e}")
            return create_compile_cache(cache_url)

    try:
        default_cache = create_compile_cache(CacheUrl.get_cache_url())
        patch_everywhere("create_compile_cache", hf_create_compile_cache, "libneuronxla")
        yield
        # The cache session ended without error
        if entry is not None:
            if isinstance(default_cache, CompileCacheS3):
                logger.warning("Skipping cache metadata update on S3 cache.")
            else:
                # Create cache entry in local cache: it can be later synchronized with the hub cache
                registry_path = default_cache.get_cache_dir_with_cache_key(REGISTRY_FOLDER)
                model_type = entry.config["model_type"]
                entry_path = f"{registry_path}/{model_type}/{entry.model_id}"
                config_path = f"{entry_path}/{entry.hash}.json"
                if not default_cache.exists(config_path):
                    oldmask = os.umask(000)
                    Path(entry_path).mkdir(parents=True, exist_ok=True)
                    os.umask(oldmask)
                    default_cache.upload_string_to_file(config_path, entry.to_json())
    finally:
        patch_everywhere("create_compile_cache", create_compile_cache, "libneuronxla")


@requires_torch_neuronx
def synchronize_hub_cache(cache_repo_id: Optional[str] = None):
    """Synchronize the neuronx compiler cache with the optimum-neuron hub cache.

    Args:
        repo_id (`Optional[str]`, default to None):
            The id of the HuggingFace cache repository, in the form 'org|user/name'.
    """
    hub_cache_proxy = _create_hub_compile_cache_proxy(cache_repo_id=cache_repo_id)
    hub_cache_proxy.synchronize()


def get_hub_cached_entries(model_id: str, cache_repo_id: Optional[str] = None):
    if cache_repo_id is None:
        cache_repo_id = get_hub_cache()
    # Allocate a Hub API with refreshed information (required for tests altering the env)
    endpoint = os.getenv("HF_ENDPOINT")
    token = get_token()
    api = HfApi(endpoint=endpoint, token=token)
    repo_files = api.list_repo_files(cache_repo_id)
    # Get the config corresponding to the model
    target_entry = ModelCacheEntry(model_id, (AutoConfig.from_pretrained(model_id)))
    # Extract model type: it will be used as primary key for lookup
    model_type = target_entry.config["model_type"]
    registry_pattern = REGISTRY_FOLDER + "/" + model_type
    model_files = [path for path in repo_files if registry_pattern in path]
    model_entries = []
    with TemporaryDirectory() as tmpdir:
        for model_path in model_files:
            local_path = api.hf_hub_download(cache_repo_id, model_path, local_dir=tmpdir)
            with open(local_path) as f:
                entry_config = json.load(f)
                # All config parameters but neuron config must match
                neuron_config = entry_config.pop("neuron")
                if entry_config == target_entry.config:
                    model_entries.append(neuron_config)
    return model_entries
