# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities for caching."""

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch_xla.core.xla_model as xm

from huggingface_hub import HfApi, HfFolder, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

from ...utils import logging
from .version_utils import get_neuronxcc_version


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PretrainedConfig


logger = logging.get_logger()


HF_API = HfApi()
HF_FOLDER = HfFolder()
HF_TOKEN = HF_FOLDER.get_token()

HASH_FILE_NAME = "pytorch_model.bin"
HF_HUB_CACHE_REPOS = ["michaelbenayoun/cache_test"]


def is_private_repo(repo_id: str) -> bool:
    HF_API.list_repo_files(repo_id=repo_id, token=HF_TOKEN)
    private = False
    try:
        HF_API.list_repo_files(repo_id=repo_id, token=False)
    except RepositoryNotFoundError:
        private = True
    return private


CUSTOM_CACHE_REPO = os.environ.get("CUSTOM_CACHE_REPO", None)
CUSTOM_CACHE_REPO_IS_PRIVATE = CUSTOM_CACHE_REPO is not None
if CUSTOM_CACHE_REPO:
    HF_HUB_CACHE_REPOS = [CUSTOM_CACHE_REPO] + HF_HUB_CACHE_REPOS
    CUSTOM_CACHE_REPO_IS_PRIVATE = is_private_repo(CUSTOM_CACHE_REPO)


NEURON_COMPILE_CACHE_NAME = "neuron-compile-cache"


def get_neuron_cache_path() -> Optional[Path]:
    neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    if "--no-cache" in neuron_cc_flags:
        return None
    else:
        match_ = re.search(r"--cache_dir=([\w\/]+)", neuron_cc_flags)
        if match_:
            path = Path(match_.group(1))
        else:
            path = Path("/var/tmp")

        return path / NEURON_COMPILE_CACHE_NAME


def set_neuron_cache_path(neuron_cache_path: Union[str, Path], ignore_no_cache: bool = False):
    neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    if "--no-cache" in neuron_cc_flags:
        if ignore_no_cache:
            neuron_cc_flags = neuron_cc_flags.replace("--no-cache", "")
        else:
            raise ValueError(
                "Cannot set the neuron compile cache since --no-cache is in NEURON_CC_FLAGS. You can overwrite this "
                "behaviour by doing ignore_no_cache=True."
            )
    if isinstance(neuron_cache_path, Path):
        neuron_cache_path = neuron_cache_path.as_posix()

    match_ = re.search(r"--cache_dir=([\w\/]+)", neuron_cc_flags)
    if match_:
        neuron_cc_flags = neuron_cc_flags[: match_.start(1)] + neuron_cache_path + neuron_cc_flags[match_.end(1) :]
    else:
        neuron_cc_flags = neuron_cc_flags + f" --cache_dir={neuron_cache_path}"

    os.environ["NEURON_CC_FLAGS"] = neuron_cc_flags


def get_num_neuron_cores_used():
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))


def list_files_in_neuron_cache(neuron_cache_path: Path, only_relevant_files: bool = False) -> List[Path]:
    files = [path for path in neuron_cache_path.glob("**/*") if path.is_file()]
    if only_relevant_files:
        files = [p for p in files if p.suffix in [".neff", ".pb", ".txt"]]
    return files


def compute_file_sha512_hash(filename: Union[str, Path]) -> str:
    if isinstance(filename, Path):
        filename = filename.as_posix()

    file_hash = hashlib.sha512()
    with open(filename, "rb") as f:
        fb = f.read()
        file_hash.update(fb)
    return file_hash.hexdigest()


class StaticTemporaryDirectory:
    def __init__(self, dirname: Union[str, Path]):
        if isinstance(dirname, str):
            dirname = Path(dirname)
        if dirname.exists():
            raise FileExistsError(
                f"{dirname} already exists, cannot create a static temporary directory witht this name."
            )
        self.dirname = dirname

    def __enter__(self):
        self.dirname.mkdir(parents=True)
        return self.dirname

    def __exit__(self, *exc):
        shutil.rmtree(self.dirname)


T = TypeVar("T")
TupleOrList = Union[Tuple[T], List[T]]


@dataclass
class _MutableHashAttribute:
    model_hash: str = ""
    overall_hash: str = ""

    @property
    def is_empty(self):
        return (not self.model_hash) or (not self.overall_hash)

    def __hash__(self):
        return hash(f"{self.model_hash}_{self.overall_hash}")


@dataclass(frozen=True)
class NeuronHash:
    model: "PreTrainedModel"
    input_shapes: Tuple[Tuple[int], ...]
    data_type: torch.dtype
    num_neuron_cores: int = field(default_factory=get_num_neuron_cores_used)
    neuron_compiler_version: str = field(default_factory=get_neuronxcc_version)
    _hash: _MutableHashAttribute = field(default_factory=_MutableHashAttribute)

    def __post_init__(self):
        self.compute_hash()

    @property
    def hash_dict(self) -> Dict[str, Any]:
       hash_dict = asdict(self)
       hash_dict["model"] = hash_dict["model"].state_dict()
       hash_dict.pop("_hash")
       return hash_dict

    def compute_hash(self) -> Tuple[str, str]:
        if self._hash.is_empty:
            model_hash = ""
            with tempfile.TemporaryDirectory() as tmpdirname:
                filename = Path(tmpdirname) / HASH_FILE_NAME
                xm.save(self.model.state_dict(), filename)
                model_hash = compute_file_sha512_hash(filename)

            overall_hash = ""
            hash_dict = self.hash_dict
            with tempfile.TemporaryDirectory() as tmpdirname:
                filename = Path(tmpdirname) / HASH_FILE_NAME
                xm.save(hash_dict, filename)
                overall_hash = compute_file_sha512_hash(filename)

            self._hash.model_hash = model_hash
            self._hash.overall_hash = overall_hash

        return self._hash.model_hash, self._hash.overall_hash

    @property
    def folders(self) -> List[str]:
        model_hash, overall_hash = self.compute_hash()
        return [
            self.neuron_compiler_version,
            self.model.config.model_type,
            model_hash,
            overall_hash,
        ]

    @property
    def cache_path(self) -> Path:
        return Path().joinpath(*self.folders)

    @property
    def neuron_compiler_version_dir_name(self):
        return f"USER_neuroncc-{self.neuron_compiler_version}"

    
    def _try_to_retrive_model_name_or_path(self, config: "PretrainedConfig") -> Optional[str]:
        attribute_names_to_try = ["_model_name_or_path", "_name_or_path"]
        model_name_or_path = None
        for name in attribute_names_to_try:
            attribute = getattr(config, name, None)
            if attribute is not None:
                model_name_or_path = attribute
                break
        return model_name_or_path

    @property
    def is_private(self):
        private = None
        model_name_or_path = self._try_to_retrive_model_name_or_path(self.model.config) 
        if model_name_or_path is None:
            private = True
        elif Path(model_name_or_path).exists():
            private = True
        else:
            private = is_private_repo(model_name_or_path)
        return private 


@dataclass
class CachedModelOnTheHub:
    repo_id: str
    folder: Union[str, Path]
    revision: str = "main"
    files_on_the_hub: str = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.folder, Path):
            self.folder = self.folder.as_posix()


def get_cached_model_on_the_hub(neuron_hash: NeuronHash) -> Optional[CachedModelOnTheHub]:
    target_directory = neuron_hash.cache_path

    cache_repo_id = None
    cache_revision = None

    for repo_id in HF_HUB_CACHE_REPOS:
        if isinstance(repo_id, tuple):
            repo_id, revision = repo_id
        else:
            revision = "main"
        repo_filenames = HfApi().list_repo_files(repo_id, revision=revision, token=HF_TOKEN)
        model_files_on_the_hub = []
        for repo_filename in repo_filenames:
            if repo_filename.startswith(target_directory.as_posix()):
                if cache_repo_id is None:
                    cache_repo_id = repo_id
                    cache_revision = revision
                model_files_on_the_hub.append(repo_filename)

    if cache_repo_id is None:
        cached_model = None
    else:
        cached_model = CachedModelOnTheHub(
            cache_repo_id, target_directory, revision=cache_revision, files_on_the_hub=model_files_on_the_hub
        )

    return cached_model


def download_cached_model_from_hub(
    neuron_hash: NeuronHash,
    target_directory: Optional[Union[str, Path]] = None,
    path_in_repo_to_path_in_target_directory: Optional[Callable[[Path], Path]] = None,
) -> bool:
    if target_directory is None:
        target_directory = get_neuron_cache_path()
        if target_directory is None:
            raise ValueError("A target directory must be specified when no caching directory is used.")
    elif isinstance(target_directory, str):
        target_directory = Path(target_directory)

    cached_model = get_cached_model_on_the_hub(neuron_hash)
    if cached_model is not None:
        folder = cached_model.folder

        ignore_patterns = []
        for filename in cached_model.files_on_the_hub:
            path_in_repo = Path(filename)
            if path_in_repo_to_path_in_target_directory is not None:
                potential_local_path = target_directory / path_in_repo_to_path_in_target_directory(path_in_repo)
            else:
                potential_local_path = target_directory / path_in_repo

            if potential_local_path.exists():
                ignore_patterns.append(filename)

        needs_to_download = cached_model.files_on_the_hub and len(ignore_patterns) != len(
            cached_model.files_on_the_hub
        )

        if needs_to_download:
            snapshot_download(
                repo_id=cached_model.repo_id,
                revision=cached_model.revision,
                repo_type="model",
                local_dir=target_directory,
                local_dir_use_symlinks=False,
                allow_patterns=f"{folder}/**",
                ignore_patterns=ignore_patterns,
                tqdm_class=None,
            )

            if path_in_repo_to_path_in_target_directory is not None:
                local_folder = target_directory / folder
                for path in local_folder.glob("**/*"):
                    if not path.is_file():
                        continue
                    target_path = target_directory / path_in_repo_to_path_in_target_directory(path)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(path, target_path)
                    # TODO: remove old path?

    return cached_model is not None


def push_to_cache_on_hub(
    neuron_hash: NeuronHash,
    local_cache_dir_or_file: Path,
    cache_repo_id: Optional[str] = None,
    overwrite_existing: bool = False,
    local_path_to_path_in_repo: Optional[Callable[[Path], Path]] = None,
) -> CachedModelOnTheHub:
    if cache_repo_id is None:
        cache_repo_id = HF_HUB_CACHE_REPOS[0]

    is_cache_repo_private = is_private_repo(cache_repo_id)
    if neuron_hash.is_private and not is_cache_repo_private:
        raise ValueError(
            f"Cannot push the cached model to {cache_repo_id} because this repo is not private but the original model is "
            "coming from private repo."
        )

    if local_path_to_path_in_repo is not None:
        path_in_repo = local_path_to_path_in_repo(local_cache_dir_or_file)
    else:
        path_in_repo = local_cache_dir_or_file

    path_in_repo = neuron_hash.cache_path / path_in_repo

    if not overwrite_existing:
        repo_filenames = map(Path, HfApi().list_repo_files(cache_repo_id, token=HF_TOKEN))
        if local_cache_dir_or_file.is_dir():
            exists = any(filename.parent == path_in_repo for filename in repo_filenames)
        else:
            exists = any(filename == path_in_repo for filename in repo_filenames)
        if exists:
            logger.info(
                f"Did not push the cached model located at {local_cache_dir_or_file} to the repo named {cache_repo_id} "
                "because it already exists there. Use overwrite_existing=True if you want to overwrite the cache on the "
                "Hub."
            )

    if local_cache_dir_or_file.is_dir():
        HF_API.upload_folder(
            folder_path=local_cache_dir_or_file.as_posix(),
            path_in_repo=path_in_repo.as_posix(),
            repo_id=cache_repo_id,
            repo_type="model",
        )
    else:
        HF_API.upload_file(
            path_or_fileobj=local_cache_dir_or_file.as_posix(),
            path_in_repo=path_in_repo.as_posix(),
            repo_id=cache_repo_id,
            repo_type="model",
        )
    return CachedModelOnTheHub(cache_repo_id, path_in_repo)
