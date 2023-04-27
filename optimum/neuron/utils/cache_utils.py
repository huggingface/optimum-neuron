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
# limitations under the License.
"""Utilities for caching."""

import hashlib
import io
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from ...utils import logging
from .version_utils import get_neuronxcc_version


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


logger = logging.get_logger()


if os.environ.get("HUGGINGFACE_CO_STAGING") == "1":
    HF_HUB_CACHE_REPOS = []
else:
    HF_HUB_CACHE_REPOS = ["aws-neuron/optimum-neuron-cache"]

HASH_FILE_NAME = "pytorch_model.bin"
NEURON_COMPILE_CACHE_NAME = "neuron-compile-cache"

_IP_PATTERN = re.compile(r"ip-([0-9]{1,3}-){4}")


def is_private_repo(repo_id: str) -> bool:
    HfApi().list_repo_files(repo_id=repo_id, token=HfFolder.get_token())
    private = False
    try:
        HfApi().list_repo_files(repo_id=repo_id, token=False)
    except RepositoryNotFoundError:
        private = True
    return private


def get_hf_hub_cache_repos():
    custom_cache_repo = os.environ.get("CUSTOM_CACHE_REPO", None)
    hf_hub_repos = HF_HUB_CACHE_REPOS
    if custom_cache_repo:
        hf_hub_repos = [custom_cache_repo] + hf_hub_repos
    return hf_hub_repos


def get_neuron_cache_path() -> Optional[Path]:
    # NEURON_CC_FLAGS is the environment variable read by the neuron compiler.
    # Among other things, this is where the cache directory is specified.
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
    # NEURON_CC_FLAGS is the environment variable read by the neuron compiler.
    # Among other things, this is where the cache directory is specified.
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


def path_after_folder(path: Path, folder: Union[str, Path], include_folder: bool = False) -> Path:
    if isinstance(folder, Path):
        folder = folder.name
    try:
        index = path.parts.index(folder)
    except ValueError:
        index = len(path.parts)
    index = index + 1 if not include_folder else index
    return Path("").joinpath(*path.parts[index:])


def remove_ip_adress_from_path(path: Path) -> Path:
    return Path().joinpath(*(re.sub(_IP_PATTERN, "", part) for part in path.parts))


class StaticTemporaryDirectory:
    def __init__(self, dirname: Union[str, Path]):
        if isinstance(dirname, str):
            dirname = Path(dirname)
        if dirname.exists():
            raise FileExistsError(
                f"{dirname} already exists, cannot create a static temporary directory with this name."
            )
        self.dirname = dirname

    def __enter__(self):
        self.dirname.mkdir(parents=True)
        return self.dirname

    def __exit__(self, *exc):
        shutil.rmtree(self.dirname)


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
        hash_dict["_model_class"] = self.model.__class__
        hash_dict["_is_model_training"] = self.model.training
        hash_dict.pop("_hash")
        return hash_dict

    def state_dict_to_bytes(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        bytes_to_join = []
        for name, tensor in state_dict.items():
            memfile = io.BytesIO()
            np.save(memfile, tensor.cpu().numpy())
            bytes_to_join.append(name.encode("utf-8"))
            bytes_to_join.append(memfile.getvalue())
        return b"".join(bytes_to_join)

    def compute_sha512_hash(self, *buffers: bytes) -> str:
        hash_ = hashlib.sha512()
        for buffer in buffers:
            hash_.update(buffer)
        return hash_.hexdigest()

    def compute_hash(self) -> Tuple[str, str]:
        if self._hash.is_empty:
            model_hash = self.compute_sha512_hash(self.state_dict_to_bytes(self.model.state_dict()))

            hash_dict = self.hash_dict
            hash_dict["model"] = model_hash
            hash_dict["data_type"] = str(hash_dict["data_type"]).split(".")[1]

            buffers = [name.encode("utf-8") + str(value).encode("utf-8") for name, value in hash_dict.items()]

            overal_hash = self.compute_sha512_hash(*buffers)
            self._hash.model_hash = model_hash
            self._hash.overall_hash = overal_hash

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
    files_on_the_hub: List[str] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.folder, Path):
            self.folder = self.folder.as_posix()


def get_cached_model_on_the_hub(neuron_hash: NeuronHash) -> Optional[CachedModelOnTheHub]:
    target_directory = neuron_hash.cache_path

    cache_repo_id = None
    cache_revision = None

    for repo_id in get_hf_hub_cache_repos():
        if isinstance(repo_id, tuple):
            repo_id, revision = repo_id
        else:
            revision = "main"
        repo_filenames = HfApi().list_repo_files(repo_id, revision=revision, token=HfFolder.get_token())
        model_files_on_the_hub = []
        was_found_in_repo = False
        for repo_filename in repo_filenames:
            if repo_filename.startswith(target_directory.as_posix()):
                if cache_repo_id is None:
                    cache_repo_id = repo_id
                    cache_revision = revision
                    was_found_in_repo = True
                model_files_on_the_hub.append(repo_filename)
        if was_found_in_repo:
            break

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

            potential_local_path = remove_ip_adress_from_path(potential_local_path)

            if potential_local_path.exists():
                ignore_patterns.append(filename)

        needs_to_download = cached_model.files_on_the_hub and len(ignore_patterns) != len(
            cached_model.files_on_the_hub
        )

        if needs_to_download:
            files_before_downloading = [f for f in (target_directory / folder).glob("**/*") if f.is_file()]
            huggingface_hub.snapshot_download(
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
                    if path.is_dir():
                        continue
                    if path in files_before_downloading:
                        continue
                    target_path = target_directory / path_in_repo_to_path_in_target_directory(path)
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(path, target_path)
                # TODO: remove old directories.

    return cached_model is not None


def push_to_cache_on_hub(
    neuron_hash: NeuronHash,
    local_cache_dir_or_file: Path,
    cache_repo_id: Optional[str] = None,
    overwrite_existing: bool = False,
    local_path_to_path_in_repo: Optional[Callable[[Path], Path]] = None,
) -> CachedModelOnTheHub:
    if cache_repo_id is None:
        cache_repo_id = get_hf_hub_cache_repos()[0]

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

    # Joining a path to a absolute path ignores the original path, so we remove the root directory "/" in this case.
    if path_in_repo.is_absolute():
        path_in_repo = Path().joinpath(*path_in_repo.parts[1:])
    path_in_repo = neuron_hash.cache_path / path_in_repo

    repo_filenames = map(Path, HfApi().list_repo_files(cache_repo_id, token=HfFolder.get_token()))
    if local_cache_dir_or_file.is_dir():
        exists = any(filename.parent == path_in_repo for filename in repo_filenames)
    else:
        exists = any(filename == path_in_repo for filename in repo_filenames)
    if exists:
        if not overwrite_existing:
            logger.info(
                f"Did not push the cached model located at {local_cache_dir_or_file} to the repo named {cache_repo_id} "
                "because it already exists there. Use overwrite_existing=True if you want to overwrite the cache on the "
                "Hub."
            )
        else:
            logger.warning(
                "Overwriting the already existing cached model on the Hub by the one located at "
                f"{local_cache_dir_or_file}"
            )

    could_not_push_message = (
        "Could not push the cached model to the repo {cache_repo_id}, most likely due to not having the write permission "
        "for this repo. Exact error: {error}."
    )
    if local_cache_dir_or_file.is_dir():
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_anynonymous_cache_dir = remove_ip_adress_from_path(
                    Path(tmpdirname) / local_cache_dir_or_file.name
                )
                shutil.copytree(local_cache_dir_or_file, local_anynonymous_cache_dir)

                for file_or_dir in sorted(local_anynonymous_cache_dir.glob("**/*"), reverse=True):
                    if file_or_dir.is_dir():
                        if not list(file_or_dir.iterdir()):
                            file_or_dir.rmdir()
                        continue
                    anonymous_file = remove_ip_adress_from_path(file_or_dir)
                    anonymous_file.parent.mkdir(parents=True, exist_ok=True)
                    if file_or_dir != anonymous_file:
                        shutil.move(file_or_dir, anonymous_file)

                HfApi().upload_folder(
                    folder_path=local_anynonymous_cache_dir.as_posix(),
                    path_in_repo=path_in_repo.as_posix(),
                    repo_id=cache_repo_id,
                    repo_type="model",
                )
        except HfHubHTTPError as e:
            # TODO: create PR when no writing rights?
            logger.warning(could_not_push_message.format(cache_repo_id=cache_repo_id, error=e))
    else:
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                local_anynonymous_cache_file = remove_ip_adress_from_path(local_cache_dir_or_file)
                if local_cache_dir_or_file != local_anynonymous_cache_file:
                    local_anynonymous_cache_file = Path(tmpdirname) / path_after_folder(
                        local_anynonymous_cache_file, NEURON_COMPILE_CACHE_NAME
                    )
                    local_anynonymous_cache_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(local_cache_dir_or_file, local_anynonymous_cache_file)

                HfApi().upload_file(
                    path_or_fileobj=local_anynonymous_cache_file.as_posix(),
                    path_in_repo=path_in_repo.as_posix(),
                    repo_id=cache_repo_id,
                    repo_type="model",
                )
        except HfHubHTTPError as e:
            # TODO: create PR when no writing rights?
            logger.warning(could_not_push_message.format(cache_repo_id=cache_repo_id, error=e))
    return CachedModelOnTheHub(cache_repo_id, path_in_repo)
