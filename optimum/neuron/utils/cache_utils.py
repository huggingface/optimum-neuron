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

import os
import re
from pathlib import Path
from uuid import uuid4

from huggingface_hub import (
    HfApi,
    RepoUrl,
    create_repo,
    get_token,
)
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from transformers import PretrainedConfig

from ...utils import logging
from ...utils.logging import warn_once
from .misc import is_main_worker, string_to_bool


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

# For testing purposes.
_DISABLE_IS_PRIVATE_REPO_CHECK: bool = string_to_bool(
    os.environ.get("OPTIMUM_NEURON_DISABLE_IS_PRIVATE_REPO_CHECK", "false")
)
if _DISABLE_IS_PRIVATE_REPO_CHECK:
    logger.warning(
        "The check that prevents you from pushing compiled files from private models is disabled. This is allowed "
        "only for testing purposes."
    )


def load_custom_cache_repo_name_from_hf_home(
    hf_home_cache_repo_file: str | Path = HF_HOME_CACHE_REPO_FILE,
) -> str | None:
    if Path(hf_home_cache_repo_file).exists():
        with open(hf_home_cache_repo_file, "r") as fp:
            repo_id = fp.read()
            return repo_id.strip()
    return None


def set_custom_cache_repo_name_in_hf_home(
    repo_id: str, hf_home: str = HF_HOME, check_repo: bool = True, api: HfApi | None = None
):
    hf_home_cache_repo_file = f"{hf_home}/{CACHE_REPO_FILENAME}"
    if api is None:
        api = HfApi()
    if check_repo:
        try:
            api.repo_info(repo_id, repo_type="model")
        except Exception as e:
            raise ValueError(
                f"Could not save the custom Neuron cache repo to be {repo_id} because it does not exist or is "
                f"private to you. Complete exception message: {e}."
            )

    existing_custom_cache_repo = load_custom_cache_repo_name_from_hf_home(hf_home_cache_repo_file)
    if is_main_worker() and existing_custom_cache_repo is not None:
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
    set_custom_cache_repo_name_in_hf_home(repo_url.repo_id)
    return repo_url


def is_private_repo(repo_id: str) -> bool:
    """Tells whether `repo_id` is private."""
    if _DISABLE_IS_PRIVATE_REPO_CHECK:
        return False
    try:
        HfApi().model_info(repo_id=repo_id, token=get_token())
        private_to_user = False
    except RepositoryNotFoundError:
        private_to_user = True
    if private_to_user:
        private = True
    else:
        try:
            HfApi().model_info(repo_id=repo_id, token=False)
            private = False
        except RepositoryNotFoundError:
            private = True
    return private


_CACHED_HAS_WRITE_ACCESS_TO_REPO = {}


def has_write_access_to_repo(repo_id: str) -> bool:
    # If the result has already been cached, use it instead of requesting the HF Hub again.
    token = get_token()
    key = (token, repo_id)
    if key in _CACHED_HAS_WRITE_ACCESS_TO_REPO:
        return _CACHED_HAS_WRITE_ACCESS_TO_REPO[key]

    api = HfApi()
    has_access = None
    try:
        api.delete_branch(repo_id=repo_id, repo_type="model", branch=f"this-branch-does-not-exist-{uuid4()}")
    except GatedRepoError:
        has_access = False
    except RepositoryNotFoundError:
        # We could raise an error to indicate the user that the repository could not even be found:
        # raise ValueError(f"Repository {repo_id} not found (repo_type: {repo_type}). Is it a private one?") from e
        # But here we simply return `False`, because it means that we do not have write access to this repo in the end.
        has_access = False
    except RevisionNotFoundError:
        has_access = True  # has write access, otherwise would have been 403 forbidden.
    except HfHubHTTPError as e:
        if e.response.status_code in (401, 403):
            has_access = False

    if has_access is None:
        raise ValueError(f"Cannot determine write access to {repo_id}")

    # Cache the result for subsequent calls.
    _CACHED_HAS_WRITE_ACCESS_TO_REPO[key] = has_access

    return has_access


def get_hf_hub_cache_repos(log_warnings: bool = False) -> list[str]:
    """
    Retrieves the name of the Hugging Face Hub model repo to use as remote cache.
    Priority:
        - If a repo is provided via the `CUSTOM_CACHE_REPO` environment variable, it will be used,
        - Else, if a custom cache repo has been set locally, it will be used,
        - Otherwise, it uses the default cache repo (on which most people do not have write access)
    """
    # Default hub repos.
    hf_hub_repos = HF_HUB_CACHE_REPOS

    # Locally saved hub repo.
    saved_custom_cache_repo = load_custom_cache_repo_name_from_hf_home()
    if saved_custom_cache_repo is not None and saved_custom_cache_repo not in hf_hub_repos:
        hf_hub_repos = [saved_custom_cache_repo] + hf_hub_repos

    # Hub repo set via the environment variable CUSTOM_CACHE_REPO.
    custom_cache_repo = os.environ.get("CUSTOM_CACHE_REPO", None)
    if custom_cache_repo is not None and custom_cache_repo not in hf_hub_repos:
        hf_hub_repos = [custom_cache_repo] + hf_hub_repos

    if log_warnings and is_main_worker() and saved_custom_cache_repo is None and custom_cache_repo is None:
        warn_once(
            logger,
            "No Neuron cache name is saved locally. This means that only the official Neuron cache will be used. You "
            "can create a Neuron cache repo by running the following command: `optimum-cli neuron cache create`. If "
            "the Neuron cache already exists you can set it by running the following command: `optimum-cli neuron cache "
            "set -n [name]`.",
        )

    if log_warnings and is_main_worker() and hf_hub_repos and not has_write_access_to_repo(hf_hub_repos[0]):
        warn_once(
            logger,
            f"You do not have write access to {hf_hub_repos[0]} so you will not be able to push any cached compilation "
            "files. Please log in and/or use a custom Neuron cache.",
        )
    return hf_hub_repos


def get_hf_hub_cache_repo(log_warnings: bool = False) -> str:
    return get_hf_hub_cache_repos(log_warnings=log_warnings)[0]


def get_neuron_cache_path() -> Path | None:
    # NEURON_CC_FLAGS is the environment variable read by the neuron compiler.
    # Among other things, this is where the cache directory is specified.
    neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    if "--no-cache" in neuron_cc_flags:
        return None
    else:
        match_ = re.search(r"--cache_dir=([\w\/-]+)", neuron_cc_flags)
        if match_:
            path = Path(match_.group(1))
        else:
            path = Path("/var/tmp/neuron-compile-cache")

        return path


def set_neuron_cache_path(neuron_cache_path: str | Path, ignore_no_cache: bool = False):
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
    neuron_devices_path = Path("/sys/class/neuron_device/")
    if not neuron_devices_path.is_dir():
        num_cores = 0
    else:
        num_cores = len(list(neuron_devices_path.iterdir())) * 2
    return num_cores


def get_num_neuron_cores_used() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def list_files_in_neuron_cache(neuron_cache_path: str | Path, only_relevant_files: bool = False) -> list[Path]:
    if isinstance(neuron_cache_path, str):
        neuron_cache_path = Path(neuron_cache_path)
    files = [path for path in neuron_cache_path.glob("**/*") if path.is_file()]
    if only_relevant_files:
        files = [p for p in files if p.suffix in [".neff", ".pb", ".txt"]]
    return files


def get_model_name_or_path(config: "PretrainedConfig") -> str | None:
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
