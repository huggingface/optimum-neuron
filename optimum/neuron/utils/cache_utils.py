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
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import huggingface_hub
import numpy as np
import torch
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    HfFolder,
    RepoUrl,
    create_repo,
    hf_hub_download,
)
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from packaging import version

from ...utils import logging
from ...utils.logging import warn_once
from .version_utils import get_neuronxcc_version


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel


logger = logging.get_logger()

HOME = Path.home()
DEFAULT_HF_HOME = f"{HOME}/.cache/huggingface"
XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME", None)
if XDG_CACHE_HOME is not None:
    DEFAULT_HF_HOME = f"{XDG_CACHE_HOME}/huggingface"
HF_HOME = os.environ.get("HF_HOME", DEFAULT_HF_HOME)

CACHE_REPO_FILENAME = "optimum_neuron_custom_cache"
HF_HOME_CACHE_REPO_FILE = f"{HF_HOME}/{CACHE_REPO_FILENAME}"

CACHE_REPO_NAME = "optimum-neuron-cache"
if os.environ.get("HUGGINGFACE_CO_STAGING") == "1":
    HF_HUB_CACHE_REPOS = []
else:
    HF_HUB_CACHE_REPOS = [f"aws-neuron/{CACHE_REPO_NAME}"]

HASH_FILENAME = "pytorch_model.bin"
REGISTRY_FILENAME = "registry.json"
NEURON_COMPILE_CACHE_NAME = "neuron-compile-cache"

_IP_PATTERN = re.compile(r"ip-([0-9]{1,3}-){4}")
_HF_HUB_HTTP_ERROR_REQUEST_ID_PATTERN = re.compile(r"\(Request ID: Root=[\w-]+\)")

_WRITING_ACCESS_CACHE: Dict[Tuple[str, str], bool] = {}
_REGISTRY_FILE_EXISTS: Dict[str, bool] = {}
_ADDED_IN_REGISTRY: Dict[Tuple[str, "NeuronHash"], bool] = {}

_NEW_CACHE_NAMING_CONVENTION_NEURONXCC_VERSION = "2.7.0.40+f7c6cf2a3"


def follows_new_cache_naming_convention(neuronxcc_version: Optional[str] = None) -> bool:
    """
    The ways the cache is handled differs starting from `_NEW_CACHE_NAMING_CONVENTION_NEURONXCC_VERSION`.
    This helper functions returns `True` if `neuronxcc_version` follows the new way the cache is handled and `False`
    otherwise.
    """
    if neuronxcc_version is None:
        neuronxcc_version = get_neuronxcc_version()
    neuronxcc_version = version.parse(neuronxcc_version)
    return neuronxcc_version >= version.parse(_NEW_CACHE_NAMING_CONVENTION_NEURONXCC_VERSION)


def load_custom_cache_repo_name_from_hf_home(
    hf_home_cache_repo_file: Union[str, Path] = HF_HOME_CACHE_REPO_FILE
) -> Optional[str]:
    if Path(hf_home_cache_repo_file).exists():
        with open(hf_home_cache_repo_file, "r") as fp:
            return fp.read()
    return None


def set_custom_cache_repo_name_in_hf_home(repo_id: str, hf_home: str = HF_HOME, check_repo: bool = True):
    hf_home_cache_repo_file = f"{hf_home}/{CACHE_REPO_FILENAME}"
    if check_repo:
        try:
            HfApi().repo_info(repo_id, repo_type="model")
        except Exception as e:
            raise ValueError(
                f"Could not save the custom Neuron cache repo to be {repo_id} because it does not exist or is "
                f"private to you. Complete exception message: {e}."
            )

    existing_custom_cache_repo = load_custom_cache_repo_name_from_hf_home(hf_home_cache_repo_file)
    if existing_custom_cache_repo is not None:
        logger.warning(
            f"A custom cache repo was already registered: {existing_custom_cache_repo}. It will be overwritten to "
            f"{repo_id}."
        )

    with open(hf_home_cache_repo_file, "w") as fp:
        fp.write(repo_id)


def delete_custom_cache_repo_name_from_hf_home(hf_home_cache_repo_file: str = HF_HOME_CACHE_REPO_FILE):
    Path(hf_home_cache_repo_file).unlink(missing_ok=True)


def create_custom_cache_repo(repo_id: str = CACHE_REPO_NAME, private: bool = True) -> RepoUrl:
    repo_url = create_repo(repo_id, private=private, repo_type="model")
    create_registry_file_if_does_not_exist(repo_id)
    set_custom_cache_repo_name_in_hf_home(repo_url.repo_id)
    return repo_url


def is_private_repo(repo_id: str) -> bool:
    HfApi().list_repo_files(repo_id=repo_id, token=HfFolder.get_token())
    private = False
    try:
        HfApi().list_repo_files(repo_id=repo_id, token=False)
    except RepositoryNotFoundError:
        private = True
    return private


def has_write_access_to_repo(repo_id: str) -> bool:
    token = HfFolder.get_token()
    if (token, repo_id) in _WRITING_ACCESS_CACHE:
        return _WRITING_ACCESS_CACHE[(token, repo_id)]

    has_access = False
    with tempfile.NamedTemporaryFile() as fp:
        tmpfilename = Path(fp.name)
        try:
            add_file = CommitOperationAdd(f"write_access_test/{tmpfilename.name}", tmpfilename.as_posix())
            HfApi().create_commit(repo_id, operations=[add_file], commit_message="Check write access")
        except (HfHubHTTPError, RepositoryNotFoundError):
            pass
        else:
            delete_file = CommitOperationDelete(f"write_access_test/{tmpfilename.name}")
            HfApi().create_commit(repo_id, operations=[delete_file], commit_message="Check write access [DONE]")
            has_access = True

    _WRITING_ACCESS_CACHE[(token, repo_id)] = has_access
    return has_access


def get_hf_hub_cache_repos():
    hf_hub_repos = HF_HUB_CACHE_REPOS

    saved_custom_cache_repo = load_custom_cache_repo_name_from_hf_home()
    if saved_custom_cache_repo is None:
        warn_once(
            logger,
            "No Neuron cache name is saved locally. This means that only the official Neuron cache, and "
            "potentially a cache defined in $CUSTOM_CACHE_REPO will be used. You can create a Neuron cache repo by "
            "running the following command: `optimum-cli neuron cache create`. If the Neuron cache already exists "
            "you can set it by running the following command: `optimum-cli neuron cache set -n [name]`.",
        )
    else:
        hf_hub_repos = [saved_custom_cache_repo] + hf_hub_repos

    custom_cache_repo = os.environ.get("CUSTOM_CACHE_REPO", None)
    if custom_cache_repo is not None:
        hf_hub_repos = [custom_cache_repo] + hf_hub_repos

    # TODO: this is a quick fix.
    # Cache utils should not be aware of the multiprocessing side of things.
    # The issue here is that `has_write_access_to_repo` actually pushes stuff to the HF Hub.
    # Pushing stuff to the HF Hub should be limited to the `push_to_cache_on_hub` function,
    # making it easier for higher-level abstractions using the cache utils to reason on which
    # parts should only run on the master process and which parts should run on everyone.
    from . import is_torch_xla_available

    process_index = 0
    if is_torch_xla_available():
        import torch_xla.core.xla_model as xm

        process_index = xm.get_ordinal()

    if process_index == 0 and hf_hub_repos and not has_write_access_to_repo(hf_hub_repos[0]):
        warn_once(
            logger,
            f"You do not have write access to {hf_hub_repos[0]} so you will not be able to push any cached compilation "
            "files. Please log in and/or use a custom Neuron cache.",
        )
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

        # TODO: is that correct?
        if not follows_new_cache_naming_convention():
            path = path / NEURON_COMPILE_CACHE_NAME

        return path


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

    match_ = re.search(r"--cache_dir=([\w\/-]+)", neuron_cc_flags)
    if match_:
        neuron_cc_flags = neuron_cc_flags[: match_.start(1)] + neuron_cache_path + neuron_cc_flags[match_.end(1) :]
    else:
        neuron_cc_flags = neuron_cc_flags + f" --cache_dir={neuron_cache_path}"

    os.environ["NEURON_CC_FLAGS"] = neuron_cc_flags


def get_num_neuron_cores() -> int:
    proc = subprocess.Popen(["neuron-ls", "-j"], stdout=subprocess.PIPE)
    stdout, _ = proc.communicate()
    stdout = stdout.decode("utf-8")
    json_stdout = json.loads(stdout)
    return json_stdout[0]["nc_count"]


def get_num_neuron_cores_used() -> int:
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


def _get_model_name_or_path(config: "PretrainedConfig") -> Optional[str]:
    attribute_names_to_try = ["_model_name_or_path", "_name_or_path"]
    model_name_or_path = None
    for name in attribute_names_to_try:
        attribute = getattr(config, name, None)
        if attribute is not None:
            model_name_or_path = attribute
            break
    if model_name_or_path == "":
        model_name_or_path = None
    return model_name_or_path


def create_registry_file_if_does_not_exist(repo_id: str):
    was_created = _REGISTRY_FILE_EXISTS.get(repo_id, False)
    if was_created:
        return
    file_exists = True
    try:
        hf_hub_download(repo_id, REGISTRY_FILENAME, force_download=True)
    except EntryNotFoundError:
        file_exists = False
    if file_exists:
        return
    with tempfile.NamedTemporaryFile() as tmpfile:
        with open(tmpfile.name, "w") as fp:
            json.dump({}, fp)
        tmpfilename = Path(tmpfile.name)
        add_registry_file = CommitOperationAdd(REGISTRY_FILENAME, tmpfilename.as_posix())
        HfApi().create_commit(repo_id, operations=[add_registry_file], commit_message="Create cache registry file")

    _REGISTRY_FILE_EXISTS[repo_id] = True


def add_in_registry(repo_id: str, neuron_hash: "NeuronHash"):
    was_added = _ADDED_IN_REGISTRY.get((repo_id, neuron_hash), False)
    if was_added:
        return
    model_name_or_path = _get_model_name_or_path(neuron_hash.model.config)
    if model_name_or_path is None:
        model_name_or_path = "null"

    model_hash, overall_hash = neuron_hash.compute_hash()

    with tempfile.TemporaryDirectory() as tmpdirname:
        keep_going = True
        while keep_going:
            tmpdirpath = Path(tmpdirname)
            head = HfApi().model_info(repo_id).sha
            hf_hub_download(
                repo_id,
                REGISTRY_FILENAME,
                revision=head,
                local_dir=tmpdirpath,
                local_dir_use_symlinks=False,
            )
            registry_path = tmpdirpath / REGISTRY_FILENAME
            with open(registry_path, "r") as fp:
                registry = json.load(fp)

            orig_registry = registry
            if neuron_hash.neuron_compiler_version not in registry:
                registry[neuron_hash.neuron_compiler_version] = {}
            registry = registry[neuron_hash.neuron_compiler_version]

            key = model_name_or_path if model_name_or_path != "null" else model_hash
            if model_name_or_path not in registry:
                registry[key] = {"model_name_or_path": model_name_or_path, "model_hash": model_hash}
            registry = registry[key]

            if "features" not in registry:
                registry["features"] = []

            exists_already = False
            for feature in registry["features"]:
                if feature["neuron_hash"] == overall_hash:
                    exists_already = True

            if not exists_already:
                data = {
                    "input_shapes": neuron_hash.input_shapes,
                    "precision": str(neuron_hash.data_type),
                    "num_neuron_cores": neuron_hash.num_neuron_cores,
                    "neuron_hash": overall_hash,
                }
                registry["features"].append(data)

            with open(registry_path, "w") as fp:
                json.dump(orig_registry, fp)

            add_model_in_registry = CommitOperationAdd(REGISTRY_FILENAME, registry_path.as_posix())
            try:
                HfApi().create_commit(
                    repo_id,
                    operations=[add_model_in_registry],
                    commit_message=f"Add {model_name_or_path} in registry for NeuronHash {overall_hash}",
                    parent_commit=head,
                )
            except ValueError as e:
                if "A commit has happened since" in str(e):
                    logger.info(
                        "A commit has happened in cache repository since we tried to update the registry, starting again..."
                    )
                else:
                    raise e
            else:
                keep_going = False

        _ADDED_IN_REGISTRY[(repo_id, neuron_hash)] = True


def _list_in_registry_dict(
    registry: Dict[str, Any],
    model_name_or_path_or_hash: Optional[str] = None,
    neuron_compiler_version: Optional[str] = None,
) -> List[str]:
    entries = []
    if neuron_compiler_version is not None:
        registry = registry.get(neuron_compiler_version, {})
    else:
        for version_ in registry:
            entries += _list_in_registry_dict(
                registry, model_name_or_path_or_hash=model_name_or_path_or_hash, neuron_compiler_version=version_
            )
        return entries

    def validate_features_input_shapes(input_shapes: Tuple[Tuple[str, Tuple[int, ...]], ...]) -> bool:
        return len(input_shapes) > 0 and all(len(entry) == 2 for entry in input_shapes)

    # model_key is either a model name or path or a model hash.
    for model_key in registry:
        data = registry[model_key]
        if model_name_or_path_or_hash is not None and not (
            data["model_name_or_path"].startswith(model_name_or_path_or_hash)
            or data["model_hash"].startswith(model_name_or_path_or_hash)
        ):
            continue

        for features in data["features"]:
            if not validate_features_input_shapes(features["input_shapes"]):
                continue
            if len(features["input_shapes"]) > 1:
                inputs = "\n\t- ".join(f"{x[0]} => {x[1]}" for x in features["input_shapes"])
                inputs = f"\t- {inputs}"
            else:
                x = features["input_shapes"][0]
                inputs = f"\t- {x[0]} => {x[1]}"
            information = [
                f"Model name:\t{data['model_name_or_path']}",
                f"Model hash:\t{data['model_hash']}",
                f"Global hash:\t{features['neuron_hash']}",
                f"Precision:\t{features['precision']}",
                f"Neuron X Compiler version:\t{neuron_compiler_version}",
                f"Num of neuron cores:\t{features['num_neuron_cores']}",
                f"Input shapes:\n{inputs}",
            ]
            entries.append("\n".join(information))
    return entries


def list_in_registry(
    repo_id: str, model_name_or_path_or_hash: Optional[str] = None, neuron_compiler_version: Optional[str] = None
):
    with tempfile.TemporaryDirectory() as tmpdirname:
        hf_hub_download(repo_id, REGISTRY_FILENAME, local_dir=tmpdirname, local_dir_use_symlinks=False)
        registry_filename = Path(tmpdirname) / REGISTRY_FILENAME
        with open(registry_filename, "r") as fp:
            registry = json.load(fp)

    return _list_in_registry_dict(
        registry,
        model_name_or_path_or_hash=model_name_or_path_or_hash,
        neuron_compiler_version=neuron_compiler_version,
    )


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
class _UnspecifiedHashAttribute:
    min_optimum_neuron_version: Optional[str] = None
    min_neuron_compiler_version: Optional[str] = None
    default: Optional[Any] = None

    @classmethod
    def with_args(
        cls,
        min_optimum_neuron_version: Optional[str] = None,
        min_neuron_compiler_version: Optional[str] = None,
        default: Optional[Any] = None,
    ) -> Callable[[], "_UnspecifiedHashAttribute"]:
        def constructor():
            return cls(
                min_optimum_neuron_version=min_optimum_neuron_version,
                min_neuron_compiler_version=min_neuron_compiler_version,
                default=default,
            )

        return constructor

    def check_requirements_are_met(self, neuron_compiler_version: str):
        if self.should_be_inserted_in_hash_dict(neuron_compiler_version) and self.default is None:
            raise ValueError("A default value must be specified.")
        # from ..version import __version__

        # optimum_neuron_requirement = True
        # if self.min_optimum_neuron_version is not None:
        #     if version.parse(__version__) >= version.parse(self.min_optimum_neuron_version):
        #         optimum_neuron_requirement = self.default is not None

        # neuron_compiler_requirement = True
        # if self.min_neuron_compiler_version is not None:
        #     if version.parse(neuron_compiler_version) >= version.parse(self.min_neuron_compiler_version):
        #         neuron_compiler_requirement = self.default is not None

        # if not optimum_neuron_requirement or not neuron_compiler_requirement:
        #     raise ValueError("A default value must be specified.")

    def should_be_inserted_in_hash_dict(self, neuron_compiler_version: str) -> bool:
        from ..version import __version__

        optimum_neuron_requirement = False
        if self.min_optimum_neuron_version is not None:
            optimum_neuron_requirement = version.parse(__version__) >= version.parse(self.min_optimum_neuron_version)

        neuron_compiler_requirement = False
        if self.min_neuron_compiler_version is not None:
            neuron_compiler_requirement = version.parse(neuron_compiler_version) >= version.parse(
                self.min_neuron_compiler_version
            )

        return optimum_neuron_requirement or neuron_compiler_requirement


@dataclass(frozen=True)
class NeuronHash:
    model: "PreTrainedModel"
    input_shapes: Tuple[Tuple[str, Tuple[int, ...]], ...]
    data_type: torch.dtype
    num_neuron_cores: int = field(default_factory=get_num_neuron_cores_used)
    neuron_compiler_version: str = field(default_factory=get_neuronxcc_version)
    fsdp: Union[int, _UnspecifiedHashAttribute] = field(
        default_factory=_UnspecifiedHashAttribute.with_args(min_optimum_neuron_version="0.0.8", default=False)
    )
    tensor_parallel_size: Union[int, _UnspecifiedHashAttribute] = field(
        default_factory=_UnspecifiedHashAttribute.with_args(min_optimum_neuron_version="0.0.8", default=1)
    )
    _hash: _MutableHashAttribute = field(default_factory=_MutableHashAttribute)

    def __post_init__(self):
        for attr in self.__dict__.values():
            if isinstance(attr, _UnspecifiedHashAttribute):
                attr.check_requirements_are_met(self.neuron_compiler_version)
        self.compute_hash()

    def _insert_potential_unspecified_hash_attribute(
        self, attribute_name: str, attribute: Any, hash_dict: Dict[str, Any]
    ):
        """
        Inserts `attribute` in `hash_dict` only if it is a specified attribute or if it has a default value.
        """
        if isinstance(attribute, _UnspecifiedHashAttribute) and attribute.should_be_inserted_in_hash_dict:
            hash_dict[attribute_name] = attribute.default
        else:
            hash_dict[attribute_name] = attribute

    @property
    def hash_dict(self) -> Dict[str, Any]:
        hash_dict = asdict(self)
        hash_dict["model"] = hash_dict["model"].state_dict()
        hash_dict["_model_class"] = self.model.__class__
        hash_dict["_is_model_training"] = self.model.training
        hash_dict.pop("_hash")

        self._insert_potential_unspecified_hash_attribute("tensor_parallel_size", self.tensor_parallel_size, hash_dict)
        self._insert_potential_unspecified_hash_attribute("fsdp", self.fsdp, hash_dict)

        return hash_dict

    def state_dict_to_bytes(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        cast_to_mapping = {
            torch.bfloat16: torch.float16,
        }
        bytes_to_join = []
        for name, tensor in state_dict.items():
            memfile = io.BytesIO()
            np.save(memfile, tensor.to(cast_to_mapping.get(tensor.dtype, tensor.dtype)).cpu().numpy())
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
        if follows_new_cache_naming_convention():
            return f"neuronxcc-{self.neuron_compiler_version}"
        return f"USER_neuroncc-{self.neuron_compiler_version}"

    @property
    def is_private(self):
        private = None
        model_name_or_path = _get_model_name_or_path(self.model.config)
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

    try:
        create_registry_file_if_does_not_exist(cache_repo_id)
        _REGISTRY_FILE_EXISTS[cache_repo_id] = True
    except HfHubHTTPError:
        pass

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
        "for this repo. Exact error:\n{error}."
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
            msg = could_not_push_message.format(cache_repo_id=cache_repo_id, error=e)
            msg = re.sub(_HF_HUB_HTTP_ERROR_REQUEST_ID_PATTERN, "", msg)
            warn_once(logger, msg)
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
            msg = could_not_push_message.format(cache_repo_id=cache_repo_id, error=e)
            msg = re.sub(_HF_HUB_HTTP_ERROR_REQUEST_ID_PATTERN, "", msg)
            warn_once(logger, msg)

    # Adding the model to the registry.
    try:
        add_in_registry(cache_repo_id, neuron_hash)
    except HfHubHTTPError:
        pass

    return CachedModelOnTheHub(cache_repo_id, path_in_repo)
