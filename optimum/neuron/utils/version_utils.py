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
"""Version utilities."""

import re
import subprocess


NEURONX_VERSION_PATTERN = re.compile(r"NeuronX Compiler version ([\w\.+]+)")


def get_neuronx_cc_version() -> str:
    proc = subprocess.Popen(["neuronx-cc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")
    match_ = re.search(NEURONX_VERSION_PATTERN, stdout)
    if match_ is None:
        match_ = re.search(NEURONX_VERSION_PATTERN, stderr)
    if match_ is None:
        raise RuntimeError("Could not infer the NeuronX Compiler version.")
    return match_.group(1)
