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
import logging
import os
from contextlib import contextmanager
from typing import Optional

from huggingface_hub import HfApi, get_token

from ..version import __version__
from .import_utils import is_neuronx_available
from .patching import patch_everywhere


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
    """A HuggingFace Hub proxy cache implementing the CompileCache API

    This cache first looks for compilation artifacts into the default cache, then the
    specified HuggingFace cache repository.

    Args:
        repo_id (`str`):
            The id of the HuggingFace cache repository, in the form 'org|user/name'.
        default_cache (`CompileCache`):
            The default neuron compiler cache (can be either a local file or S3 cache).
        endpoint (`Optional[str]`):
            The HuggingFaceHub endpoint: only required for unit tests to switch to the staging Hub.
        token (`Optional[str]`):
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

    def get_cache_dir(self, model_hash, compile_flags_str):
        return self.default_cache.get_cache_dir(model_hash, compile_flags_str)

    def clean(self):
        # Should we clean the Hf Hub cache also ?
        self.default_cache.clean()

    def clear_locks(self):
        # Clear locks in the default cache only, as the Hf already deals with concurrency
        self.default_cache.clear_locks()

    def get_hlos(self, failed_neff_str=""):
        return self.default_cache.get_hlos(failed_neff_str)

    def hlo_acquire_lock(self, h):
        # Put a lock in the default cache only, as the Hf already deals with concurrency
        return self.default_cache.hlo_acquire_lock(h)

    def hlo_release_lock(self, h):
        # Release lock in the default cache only, as the Hf already deals with concurrency
        return self.default_cache.hlo_release_lock(h)

    def remove(self, path):
        # Only remove in the default cache
        return self.default_cache.remove(path)

    def _rel_path(self, path):
        # Remove the default cache url from the path
        if path.startswith(self.default_cache.cache_path):
            return path[len(self.default_cache.cache_path) :].lstrip("/")

    def exists(self, path):
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

    def download_file(self, filename, dst_path):
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
            raise ValueError("HuggingFace hub compiler cache synchronization is not supported for S3.")
        logger.info(f"Synchronizing {self.repo_id} Hub cache with {self.default_cache.cache_path} local cache")
        self.api.upload_folder(
            repo_id=self.repo_id,
            folder_path=self.default_cache.cache_path,
            commit_message="Synchronizing local compiler cache.",
            ignore_patterns="lock",
        )
        logger.info("Synchronization complete.")

    def upload_file(self, cache_path, src_path):
        # Only upload to the default cache: use synchronize to populate the Hub cache
        self.default_cache.upload_file(cache_path, src_path)

    def upload_string_to_file(self, cache_path, data):
        # Only upload to the default cache: use synchronize to populate the Hub cache
        self.default_cache.upload_string_to_file(cache_path, data)

    def download_file_to_string(self, filename, limit=None):
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


@contextmanager
def hub_neuronx_cache():
    """A context manager to trigger the HuggingFace Hub proxy compiler cache"""
    if not is_neuronx_available():
        raise ImportError("Neuronx compiler is not avilable: please reinstall optimum-neuron[neuronx]")

    def hf_create_compile_cache(cache_url):
        try:
            return _create_hub_compile_cache_proxy(cache_url)
        except Exception as e:
            logger.warning(f"Bypassing Hub cache because of the following error: {e}")
            return create_compile_cache(cache_url)

    try:
        patch_everywhere("create_compile_cache", hf_create_compile_cache, "libneuronxla")
        yield
    finally:
        patch_everywhere("create_compile_cache", create_compile_cache, "libneuronxla")


def synchronize_hub_cache(cache_repo_id: Optional[str] = None):
    """Synchronize the neuronx compiler cache with the optimum-neuron hub cache.

    Args:
        repo_id (`Optional[str]`):
            The id of the HuggingFace cache repository, in the form 'org|user/name'.
    """
    if not is_neuronx_available():
        raise ImportError("Neuronx compiler is not avilable: please reinstall optimum-neuron[neuronx]")
    hub_cache_proxy = _create_hub_compile_cache_proxy(cache_repo_id=cache_repo_id)
    hub_cache_proxy.synchronize()
