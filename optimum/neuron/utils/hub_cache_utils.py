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
import copy
import hashlib
import json
import logging
import os
import shutil
from contextlib import contextmanager, nullcontext
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Literal, Optional, Union

from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile
from transformers import AutoConfig, PretrainedConfig

from ..version import __version__
from .cache_utils import get_hf_hub_cache_repo, get_neuron_cache_path
from .import_utils import is_neuronx_available
from .patching import patch_everywhere
from .require_utils import requires_torch_neuronx
from .version_utils import get_neuronxcc_version


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

CACHE_WHITE_LIST = [
    "_name_or_path",
    "transformers_version",
    "_diffusers_version",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
    "torchscript",
    "torch_dtype",
    "_commit_hash",
    "sample_size",
    "projection_dim",
    "_use_default_values",
]
NEURON_CONFIG_WHITE_LIST = ["input_names", "output_names", "model_type"]

DEFAULT_PATH_FOR_NEURON_CC_WRAPPER = Path(__file__).parent.as_posix()


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
    cache_url: Optional[CacheUrl] = None,
    cache_repo_id: Optional[str] = None,
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


class ModelCacheEntry:
    """A class describing a model cache entry

    Args:
        model_id (`str`):
            The model id, used as a key for the cache entry.
        config (`transformers.PretrainedConfig`):
            The configuration of the model.

    """

    def __init__(self, model_id: str, config: Union[PretrainedConfig, Dict[str, Any]]):
        self.model_id = model_id
        # Remove keys set to default values
        self.config = config.to_diff_dict() if isinstance(config, PretrainedConfig) else dict(config)
        excluded_keys = ["_name_or_path", "transformers_version"]
        for key in excluded_keys:
            self.config.pop(key, None)

    def to_json(self) -> str:
        return json.dumps(self.config, sort_keys=True)

    @property
    def hash(self):
        hash_gen = hashlib.sha512()
        hash_gen.update(self.to_json().encode("utf-8"))
        return str(hash_gen.hexdigest())[:20]


REGISTRY_FOLDER = f"0_REGISTRY/{__version__}"


class Mode(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"


def get_registry_folder_for_mode(mode: Union[Literal["training"], Literal["inference"], Mode]) -> str:
    if isinstance(mode, str) and not isinstance(mode, Mode):
        mode = Mode(mode)
    if mode is Mode.TRAINING:
        return f"{REGISTRY_FOLDER}/training"
    else:
        return f"{REGISTRY_FOLDER}/inference"


@requires_torch_neuronx
@contextmanager
def hub_neuronx_cache(
    mode: Union[Literal["training"], Literal["inference"], Mode],
    entry: Optional[ModelCacheEntry] = None,
    cache_repo_id: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
):
    """A context manager to activate the Hugging Face Hub proxy compiler cache.

    Args:
        mode (`Union[Literal["training"], Literal["inference"], Mode]`):
            The mode in which the context manager is used. Can be either "training" or "inference".
            This information will be used to populate the proper registry folder.
        entry (`Optional[ModelCacheEntry]`, defaults to `None`):
            An optional dataclass containing metadata associated with the model corresponding
            to the cache session. Will create a dedicated entry in the cache registry.
        cache_repo_id (`Optional[str]`, defaults to `None`):
            The id of the cache repo to use to fetch the precompiled files.
        cache_dir (`Optional[Union[str, Path]]`, defaults to `None`):
            The directory that is used as local cache directory.
    """
    registry_folder = get_registry_folder_for_mode(mode)

    def hf_create_compile_cache(cache_url):
        try:
            return create_hub_compile_cache_proxy(cache_url, cache_repo_id=cache_repo_id)
        except Exception as e:
            logger.warning(f"Bypassing Hub cache because of the following error: {e}")
            return create_compile_cache(cache_url)

    try:
        if mode == "training" and cache_dir is None:
            cache_dir = get_neuron_cache_path()
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
                registry_path = default_cache.get_cache_dir_with_cache_key(registry_folder)
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


@contextmanager
def patch_neuron_cc_wrapper(
    directory: Optional[Union[str, Path]] = DEFAULT_PATH_FOR_NEURON_CC_WRAPPER, restore_path: bool = True
):
    """
    Patches the `neuron_cc_wrapper` file to force it use our own version of it which essentially makes sure that it
    uses our caching system.
    """
    context_manager = TemporaryDirectory() if directory is None else nullcontext(enter_result=directory)
    tmpdirname = ""
    try:
        with context_manager as dirname:
            tmpdirname = dirname
            src = Path(__file__).parent / "neuron_cc_wrapper"
            dst = Path(tmpdirname) / "neuron_cc_wrapper"
            if src != dst:
                shutil.copy(src, dst)

            path = os.environ["PATH"]
            os.environ["PATH"] = f"{tmpdirname}:{path}"

            yield
    except Exception as e:
        raise e
    finally:
        if restore_path:
            os.environ["PATH"] = os.environ["PATH"].replace(f"{tmpdirname}:", "")


@requires_torch_neuronx
def synchronize_hub_cache(cache_path: Optional[Union[str, Path]] = None, cache_repo_id: Optional[str] = None):
    """Synchronize the neuronx compiler cache with the optimum-neuron hub cache.

    Args:
        cache_path (`Optional[Union[str, Path]]`, defaults to `None`):
            The path of the folder to use for synchronization.
        cache_repo_id (`Optional[str]`, default to None):
            The id of the HuggingFace cache repository, in the form 'org|user/name'.
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
    hub_cache_proxy.synchronize()


def get_hub_cached_entries(
    model_id: str, mode: Union[Literal["training"], Literal["inference"], Mode], cache_repo_id: Optional[str] = None
):
    if cache_repo_id is None:
        cache_repo_id = get_hf_hub_cache_repo()
    # Allocate a Hub API with refreshed information (required for tests altering the env)
    endpoint = os.getenv("HF_ENDPOINT")
    token = get_token()
    api = HfApi(endpoint=endpoint, token=token)
    repo_files = api.list_repo_files(cache_repo_id)
    # Get the config corresponding to the model
    try:
        config = AutoConfig.from_pretrained(model_id)
    except Exception:
        config = get_multimodels_configs_from_hub(model_id)  # Applied on SD, encoder-decoder models
    target_entry = ModelCacheEntry(model_id, config)
    # Extract model type: it will be used as primary key for lookup
    model_type = target_entry.config["model_type"]
    registry_folder = get_registry_folder_for_mode(mode)
    registry_pattern = registry_folder + "/" + model_type
    model_files = [path for path in repo_files if registry_pattern in path]
    white_list = CACHE_WHITE_LIST  # All parameters except those in the whitelist must match
    model_entries = []
    with TemporaryDirectory() as tmpdir:
        for model_path in model_files:
            local_path = api.hf_hub_download(cache_repo_id, model_path, local_dir=tmpdir)
            with open(local_path) as f:
                entry_config = json.load(f)
                if entry_config:
                    model_entries = lookup_matched_entries(
                        entry_config, target_entry, white_list, model_entries, model_type
                    )

    return model_entries


def get_hub_cached_models(
    mode: Union[Literal["training"], Literal["inference"], Mode], cache_repo_id: Optional[str] = None
):
    """Get the list of cached models for the specified mode for the current version

    Args:
        mode (`Union[Literal["training"], Literal["inference"], Mode]`): the cache mode (inference or training).
        cache_repo_id (`Optional[str]`): the path to a cache repo id if different from the default one.
    Returns:
        A set of (model_arch, model_org, model_id)
    """
    if cache_repo_id is None:
        cache_repo_id = get_hf_hub_cache_repo()
    registry_folder = get_registry_folder_for_mode(mode)
    api = HfApi()
    root = api.list_repo_tree(cache_repo_id, path_in_repo="", recursive=False)
    for root_file in root:
        compiler_pattern = "neuronxcc-"
        if is_neuronx_available():
            # If we know the current compiler we can avoid going through all of them in the hub cache
            compiler_pattern += get_neuronxcc_version()
        if root_file.path.startswith(compiler_pattern):
            # Look for a registry of cached models for the current optimum-version
            path_in_repo = root_file.path + "/" + registry_folder
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


def _prepare_config_for_matching(entry_config: Dict, target_entry: ModelCacheEntry, model_type: str):
    if model_type == "stable-diffusion":
        # Remove neuron config for comparison as the target does not have it
        neuron_config = entry_config["unet"].pop("neuron")
        non_checked_components = [
            "vae",
            "vae_encoder",
            "vae_decoder",
        ]  # Exclude vae configs from the check for now since it's complex and not mandatory
        for param in non_checked_components:
            entry_config.pop(param, None)
            target_entry.config.pop(param, None)
        target_entry_config = target_entry.config
    else:
        # Remove neuron config for comparison as the target does not have it
        neuron_config = entry_config.pop("neuron")
        entry_config = {"model": entry_config}
        target_entry_config = {"model": target_entry.config}

    return entry_config, target_entry_config, neuron_config


def lookup_matched_entries(entry_config, target_entry, white_list, model_entries, model_type: str):
    is_matched = True
    entry_config, target_entry_config, neuron_config = _prepare_config_for_matching(
        entry_config, target_entry, model_type
    )
    for name, value in entry_config.items():
        if isinstance(value, dict):
            for param in white_list:
                value.pop(param, None)
                target_entry_config[name].pop(param, None)
            for term in set(entry_config[name]).intersection(set(target_entry_config[name])):
                if entry_config[name][term] != target_entry_config[name][term]:
                    is_matched = False
                    break
        else:
            if value != target_entry_config[name]:
                is_matched = False
                break
    if is_matched:
        neuron_config.pop("model_type", None)
        model_entries.append(neuron_config)

    return model_entries


def get_multimodels_configs_from_hub(model_id):
    api = HfApi()
    repo_files = api.list_repo_files(model_id)
    config_pattern = "/config.json"
    config_files = [path for path in repo_files if config_pattern in path]
    lookup_configs = {}
    with TemporaryDirectory() as tmpdir:
        for model_path in config_files:
            local_path = api.hf_hub_download(model_id, model_path, local_dir=tmpdir)
            with open(local_path) as f:
                entry_config = json.load(f)
                white_list = CACHE_WHITE_LIST
                for param in white_list:
                    entry_config.pop(param, None)
                lookup_configs[model_path.split("/")[-2]] = entry_config

    if "unet" in lookup_configs:
        lookup_configs["model_type"] = "stable-diffusion"
    if "transformer" in lookup_configs:
        lookup_configs["model_type"] = "diffusion-transformer"
    return lookup_configs


def exclude_white_list_from_config(
    config: Dict, white_list: Optional[List] = None, neuron_white_list: Optional[List] = None
):
    if white_list is None:
        white_list = CACHE_WHITE_LIST

    if neuron_white_list is None:
        neuron_white_list = NEURON_CONFIG_WHITE_LIST

    for param in white_list:
        config.pop(param, None)

    for param in neuron_white_list:
        config["neuron"].pop(param, None)

    return config


def build_cache_config(
    configs: Union[PretrainedConfig, Dict[str, PretrainedConfig]],
    white_list: Optional[List] = None,
    neuron_white_list: Optional[List] = None,
):
    """Only applied on traced TorchScript models."""
    clean_configs = {}
    no_check_components = [
        "vae",
        "vae_encoder",
        "vae_decoder",
    ]  # Exclude vae configs from stable diffusion pipeline since it's complex and not mandatory
    if isinstance(configs, PretrainedConfig):
        configs = {"model": configs}
    for name, config in configs.items():
        if name in no_check_components:
            continue
        config = copy.deepcopy(config).to_diff_dict() if isinstance(config, PretrainedConfig) else config
        config = exclude_white_list_from_config(config, white_list, neuron_white_list)
        clean_configs[name] = config

    if len(clean_configs) > 1:
        if "unet" in configs:
            # stable diffusion
            clean_configs["model_type"] = "stable-diffusion"
        elif "transformer" in configs:
            # diffusion transformer
            clean_configs["model_type"] = "diffusion-transformer"
        else:
            # seq-to-seq
            clean_configs["model_type"] = next(iter(clean_configs.values()))["model_type"]

        return clean_configs
    else:
        return next(iter(clean_configs.values()))


def cache_traced_neuron_artifacts(neuron_dir: Path, cache_entry: ModelCacheEntry):
    # Use the context manager just for creating registry, AOT compilation won't leverage `create_compile_cache`
    # in `libneuronxla`, so we will need to cache compiled artifacts to local manually.
    with hub_neuronx_cache("inference", entry=cache_entry):
        compile_cache = create_hub_compile_cache_proxy()
        model_cache_dir = compile_cache.default_cache.get_cache_dir_with_cache_key(f"MODULE_{cache_entry.hash}")
        compile_cache.upload_folder(cache_dir=model_cache_dir, src_dir=neuron_dir)

        logger.info(f"Model cached in: {model_cache_dir}.")
