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
import os
import socket
from tempfile import TemporaryDirectory
from time import time

import pytest
from huggingface_hub import HfApi


@pytest.fixture
def cache_repos():
    # Setup: create temporary Hub repository and local cache directory
    api = HfApi()
    hostname = socket.gethostname()
    cache_repo_id = f"{hostname}-{time()}-optimum-neuron-cache"
    cache_repo_id = api.create_repo(cache_repo_id, private=True).repo_id
    cache_dir = TemporaryDirectory()
    cache_path = cache_dir.name
    # Modify environment to force neuronx cache to use temporary caches
    previous_env = {}
    env_vars = ["NEURON_COMPILE_CACHE_URL", "CUSTOM_CACHE_REPO"]
    for var in env_vars:
        previous_env[var] = os.environ.get(var)
    os.environ["NEURON_COMPILE_CACHE_URL"] = cache_path
    os.environ["CUSTOM_CACHE_REPO"] = cache_repo_id
    try:
        yield (cache_path, cache_repo_id)
    finally:
        # Teardown
        api.delete_repo(cache_repo_id)
        for var in env_vars:
            if previous_env[var] is None:
                os.environ.pop(var)
            else:
                os.environ[var] = previous_env[var]
