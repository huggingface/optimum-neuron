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

from libneuronxla.neuron_cc_wrapper import main as neuron_cc_wrapper_main

from .cache_utils import get_hf_hub_cache_repos, get_neuron_cache_path
from .hub_cache_utils import hub_neuronx_cache


def main():
    with hub_neuronx_cache("training", cache_repo_id=get_hf_hub_cache_repos()[0], cache_dir=get_neuron_cache_path()):
        return neuron_cc_wrapper_main()


if __name__ == "__main__":
    main()
