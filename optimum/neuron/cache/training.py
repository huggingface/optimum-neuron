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
from contextlib import contextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory


logger = logging.getLogger(__name__)

DEFAULT_PATH_FOR_NEURON_CC_WRAPPER = Path(__file__).parent.as_posix()


@contextmanager
def patch_neuron_cc_wrapper(
    directory: str | Path | None = DEFAULT_PATH_FOR_NEURON_CC_WRAPPER, restore_path: bool = True
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
