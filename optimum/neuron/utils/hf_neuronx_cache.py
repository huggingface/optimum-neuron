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
import io
import os
from contextlib import contextmanager
from typing import Optional

import libneuronxla
from huggingface_hub import HfApi
from libneuronxla.neuron_cc_cache import CompileCache, CompileCacheFs, CompileCacheS3

from ..version import __version__
from .patching import patch_everywhere


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
            The HuggingFaceHub endpoint: only required for unit tests to swith to the staging Hub.
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
        if not self.api.repo_exists(repo_id, token=token):
            raise ValueError(f"The {repo_id} repository does not exist or you don't have access to it.")
        self.repo_id = repo_id
        self.token = token

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
        return self.api.file_exists(self.repo_id, self._rel_path(path), token=self.token)

    def download_file(self, filename, dst_path):
        # Always prioritize the default cache for faster retrieval
        if self.default_cache.exists(filename):
            self.default_cache.download_file(filename, dst_path)
        else:
            local_path = self.api.hf_hub_download(self.repo_id, self._rel_path(filename))
            os.symlink(local_path, dst_path)

    def upload_file(self, cache_path, src_path):
        # Always upload first to the default cache for faster retrieval
        self.default_cache.upload_file(cache_path, src_path)
        try:
            # Try to upload to the Hf cache
            path_in_repo = self._rel_path(cache_path)
            self.api.upload_file(
                path_or_fileobj=src_path, path_in_repo=path_in_repo, repo_id=self.repo_id, token=self.token
            )
        except Exception as e:
            print(f"Upload failed for {path_in_repo}. {e}")

    def upload_string_to_file(self, cache_path, data):
        # Always upload first to the default cache for faster retrieval
        self.default_cache.upload_string_to_file(cache_path, data)
        # Try to upload to the Hf cache
        try:
            # The HfHub library only accepts binary objects for upload
            fileobj = io.BytesIO()
            fileobj.write(data.encode())
            fileobj.seek(0)
            path_in_repo = self._rel_path(cache_path)
            self.api.upload_file(
                path_or_fileobj=fileobj, path_in_repo=path_in_repo, repo_id=self.repo_id, token=self.token
            )
        except Exception as e:
            print(f"Upload failed for {path_in_repo}. {e}")

    def download_file_to_string(self, filename, limit=None):
        # Always prioritize the default cache for faster retrieval
        if self.default_cache.exists(filename):
            return self.default_cache.download_file_to_string(filename, limit)
        local_path = self.api.hf_hub_download(self.repo_id, self._rel_path(filename))
        with open(local_path, "rb") as f:
            s = f.read().decode(errors="replace")
        return s


@contextmanager
def hf_neuronx_cache(hf_cache_id: Optional[str] = None, endpoint: Optional[str] = None, token: Optional[str] = None):
    """A context manager to trigger the HuggingFace Hub proxy compiler cache


    Args:
        hf_cache_id (`str`):
            The id of the HuggingFace cache repository, in the form 'org|user/name'.
        endpoint (`Optional[str]`):
            The HuggingFaceHub endpoint: only required for unit tests to swith to the staging Hub.
        token (`Optional[str]`):
            The HuggingFace token to use to fetch/push artifacts. If not specified it will correspond
            to the current user token.
    """
    if hf_cache_id is None:
        hf_cache_id = "dacorvo/optimum-neuron-cache"

    def hf_create_compile_cache(cache_url):
        default_cache = CompileCacheS3(cache_url) if cache_url.is_s3() else CompileCacheFs(cache_url)
        return CompileCacheHfProxy(hf_cache_id, default_cache, endpoint=endpoint, token=token)

    try:
        create_compile_cache = libneuronxla.neuron_cc_cache.create_compile_cache
        patch_everywhere("create_compile_cache", hf_create_compile_cache, "libneuronxla")
        yield
    finally:
        patch_everywhere("create_compile_cache", create_compile_cache, "libneuronxla")
