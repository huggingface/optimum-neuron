# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os
from argparse import ArgumentParser

from huggingface_hub import login

from optimum.neuron.utils.hub_neuronx_cache import synchronize_hub_cache

from .utils import OPTIMUM_INTERNAL_TESTING_CACHE_REPO_FOR_CI


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="The cache directory that contains the compilation files."
    )
    return parser.parse_args()


def run(args):
    token = os.environ.get("HF_TOKEN", None)
    login(token)
    synchronize_hub_cache(cache_path=args.cache_dir, cache_repo_id=OPTIMUM_INTERNAL_TESTING_CACHE_REPO_FOR_CI)


if __name__ == "__main__":
    args = get_args()
    run(args)
