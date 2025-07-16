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
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile

from optimum.exporters import TasksManager

from ..utils.cache_utils import get_hf_hub_cache_repo
from ..utils.import_utils import is_neuronx_available
from ..utils.patching import patch_everywhere
from ..utils.require_utils import requires_torch_neuronx
from ..utils.version_utils import get_neuronxcc_version
from ..version import __version__
from .entries.cache_entry import ModelCacheEntry


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
        endpoint (`str | None`, defaults to None):
            The HuggingFaceHub endpoint: only required for unit tests to switch to the staging Hub.
        token (`str | None`, defaults to None):
            The HuggingFace token to use to fetch/push artifacts. If not specified it will correspond
            to the current user token.
    """

    cache_type = "hf"

    def __init__(
        self, repo_id: str, default_cache: CompileCache, endpoint: str | None = None, token: str | None = None
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

    def download_folder(self, folder_path: str, dst_path: str):
        # Always prioritize the default cache for faster retrieval
        if self.default_cache.exists(folder_path):
            # cached locally
            return True
        else:
            # cached remotely
            rel_folder_path = self._rel_path(folder_path)
            try:
                folder_info = list(self.api.list_repo_tree(self.repo_id, rel_folder_path, recursive=True))
                folder_exists = len(folder_info) > 1
            except Exception as e:
                logger.info(f"{rel_folder_path} not found in {self.repo_id}: {e} \nThe model will be recompiled.")
                folder_exists = False

            if folder_exists:
                try:
                    for repo_content in folder_info:
                        if isinstance(repo_content, RepoFile):
                            local_path = self.api.hf_hub_download(self.repo_id, repo_content.path)
                            new_dst_path = Path(dst_path) / repo_content.path.split(Path(dst_path).name + "/")[-1]
                            new_dst_path.parent.mkdir(parents=True, exist_ok=True)
                            os.symlink(local_path, new_dst_path)

                    logger.info(f"Fetched cached {rel_folder_path} from {self.repo_id}")
                except Exception as e:
                    logger.warning(
                        f"Unable to download cached model in {self.repo_id}: {e} \nThe model will be recompiled."
                    )
                    folder_exists = False

            return folder_exists

    def synchronize(self, non_blocking: bool = False):
        if isinstance(self.default_cache, CompileCacheS3):
            raise ValueError("Hugging Face hub compiler cache synchronization is not supported for S3.")
        logger.info(f"Synchronizing {self.repo_id} Hub cache with {self.default_cache.cache_path} local cache")

        if os.environ.get("TORCHELASTIC_RUN_ID", None) is None:
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=self.default_cache.cache_path,
                commit_message="Synchronizing local compiler cache.",
                ignore_patterns="lock",
                run_as_future=non_blocking,
            )
        else:
            if xr.local_ordinal() == 0:
                # Only the first process uploads the cache to the Hub
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=self.default_cache.cache_path,
                    commit_message="Synchronizing local compiler cache.",
                    ignore_patterns="lock",
                    run_as_future=non_blocking,
                )
            xm.rendezvous("synchronize_hub_cache")
        logger.info("Synchronization complete.")

    def upload_file(self, cache_path: str, src_path: str):
        # Only upload to the default cache: use synchronize to populate the Hub cache
        self.default_cache.upload_file(cache_path, src_path)

    def upload_folder(self, cache_dir: str, src_dir: str):
        # Upload folder to the default cache: use synchronize to populate the Hub cache
        shutil.copytree(src_dir, cache_dir, dirs_exist_ok=True)

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


def create_hub_compile_cache_proxy(
    cache_url: CacheUrl | None = None,
    cache_repo_id: str | None = None,
):
    if cache_url is None:
        cache_url = CacheUrl.get_cache_url()
    if cache_repo_id is None:
        cache_repo_id = get_hf_hub_cache_repo()
    default_cache = CompileCacheS3(cache_url) if cache_url.is_s3() else CompileCacheFs(cache_url)
    # Reevaluate endpoint and token (needed for tests altering the environment)
    endpoint = os.getenv("HF_ENDPOINT")
    token = get_token()
    return CompileCacheHfProxy(cache_repo_id, default_cache, endpoint=endpoint, token=token)


REGISTRY_FOLDER = f"0_REGISTRY/{__version__}"


@requires_torch_neuronx
@contextmanager
def hub_neuronx_cache(
    entry: ModelCacheEntry | None = None,
    cache_repo_id: str | None = None,
    cache_dir: str | Path | None = None,
):
    """A context manager to activate the Hugging Face Hub proxy compiler cache.

    Args:
        entry (`ModelCacheEntry | None`, defaults to `None`):
            An optional dataclass containing metadata associated with the model corresponding
            to the cache session. Will create a dedicated entry in the cache registry.
        cache_repo_id (`str | None`, defaults to `None`):
            The id of the cache repo to use to fetch the precompiled files.
        cache_dir (`str | Path | None`, defaults to `None`):
            The directory that is used as local cache directory.
    """

    def hf_create_compile_cache(cache_url):
        try:
            return create_hub_compile_cache_proxy(cache_url, cache_repo_id=cache_repo_id)
        except Exception as e:
            logger.warning(f"Bypassing Hub cache because of the following error: {e}")
            return create_compile_cache(cache_url)

    try:
        if isinstance(cache_dir, Path):
            cache_dir = cache_dir.as_posix()
        default_cache = create_compile_cache(CacheUrl.get_cache_url(cache_dir=cache_dir))
        patch_everywhere("create_compile_cache", hf_create_compile_cache, "libneuronxla")
        yield
        # The cache session ended without error
        if entry is not None:
            if isinstance(default_cache, CompileCacheS3):
                logger.warning("Skipping cache metadata update on S3 cache.")
            else:
                # Create cache entry in local cache: it can be later synchronized with the hub cache
                registry_path = default_cache.get_cache_dir_with_cache_key(REGISTRY_FOLDER)
                entry_path = f"{registry_path}/{entry.model_type}/{entry.model_id}"
                config_path = f"{entry_path}/{entry.hash}.json"
                if not default_cache.exists(config_path):
                    oldmask = os.umask(000)
                    Path(entry_path).mkdir(parents=True, exist_ok=True)
                    os.umask(oldmask)
                    default_cache.upload_string_to_file(config_path, entry.serialize())
    finally:
        patch_everywhere("create_compile_cache", create_compile_cache, "libneuronxla")


@requires_torch_neuronx
def synchronize_hub_cache(
    cache_path: str | Path | None = None, cache_repo_id: str | None = None, non_blocking: bool = False
):
    """Synchronize the neuronx compiler cache with the optimum-neuron hub cache.

    Args:
        cache_path (`str | Path | None`, defaults to `None`):
            The path of the folder to use for synchronization.
        cache_repo_id (`str | None`, defaults to `None`):
            The id of the HuggingFace cache repository, in the form 'org|user/name'.
        non_blocking (`bool`, defaults to `False`):
            If `True`, the synchronization will be done in a non-blocking way, i.e. the function will return immediately
            and the synchronization will be done in the background.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path_str = cache_path.as_posix()
        if not cache_path.is_dir():
            raise ValueError(f"The {cache_path_str} directory does not exist, cannot synchronize.")
        cache_url = CacheUrl(cache_path_str, url_type="fs")
    else:
        cache_url = None
    hub_cache_proxy = create_hub_compile_cache_proxy(cache_url=cache_url, cache_repo_id=cache_repo_id)
    hub_cache_proxy.synchronize(non_blocking=non_blocking)


def get_hub_cached_entries(
    model_id: str,
    task: str | None = None,
    cache_repo_id: str | None = None,
):
    if task is None:
        # Infer task from model_id
        task = TasksManager.infer_task_from_model(model_id)
    if cache_repo_id is None:
        cache_repo_id = get_hf_hub_cache_repo()
    # Allocate a Hub API with refreshed information (required for tests altering the env)
    endpoint = os.getenv("HF_ENDPOINT")
    token = get_token()
    api = HfApi(endpoint=endpoint, token=token)
    repo_files = api.list_repo_files(cache_repo_id)
    # Get the config corresponding to the model
    target_entry = ModelCacheEntry.create(model_id, task)
    registry_pattern = REGISTRY_FOLDER + "/" + target_entry.model_type
    model_files = [path for path in repo_files if registry_pattern in path]
    model_entries = []
    with TemporaryDirectory() as tmpdir:
        for model_path in model_files:
            local_path = api.hf_hub_download(cache_repo_id, model_path, local_dir=tmpdir)
            with open(local_path) as f:
                entry = ModelCacheEntry.deserialize(f.read())
                if entry.has_same_arch(target_entry):
                    model_entries.append(entry.neuron_config)
    return model_entries


def get_hub_cached_models(cache_repo_id: str | None = None):
    """Get the list of cached models for the specified mode for the current version

    Args:
        cache_repo_id (`str | None`): the path to a cache repo id if different from the default one.
    Returns:
        A set of (model_arch, model_org, model_id)
    """
    if cache_repo_id is None:
        cache_repo_id = get_hf_hub_cache_repo()
    api = HfApi()
    root = api.list_repo_tree(cache_repo_id, path_in_repo="", recursive=False)
    for root_file in root:
        compiler_pattern = "neuronxcc-"
        if is_neuronx_available():
            # If we know the current compiler we can avoid going through all of them in the hub cache
            compiler_pattern += get_neuronxcc_version()
        if root_file.path.startswith(compiler_pattern):
            # Look for a registry of cached models for the current optimum-version
            path_in_repo = root_file.path + "/" + REGISTRY_FOLDER
            root_sub_paths = path_in_repo.split("/")
            try:
                registry = api.list_repo_tree(cache_repo_id, path_in_repo=path_in_repo, recursive=True)
                cached_models = set()
                for registry_file in registry:
                    # Extract each cached model as a tuple of (arch, org, model)
                    if registry_file.path.endswith(".json"):
                        sub_paths = registry_file.path.split("/")
                        if len(sub_paths) == len(root_sub_paths) + 4:
                            # Look at the last four splits, i.e. model_arch/model_org/model_name/SHA.json
                            model_arch, model_org, model_name = sub_paths[-4:-1]
                            cached_models.add((model_arch, model_org, model_name))
                return cached_models
            except EntryNotFoundError:
                # No cached models for the current version
                continue
    return set()
