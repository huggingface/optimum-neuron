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
"""Local Neuron compile cache utilities.

Bucket-based remote cache utilities live in optimum.neuron.cache.bucket_utils.
"""

import os
import re
from pathlib import Path

from transformers import PretrainedConfig

from ...utils import logging


logger = logging.get_logger()


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
