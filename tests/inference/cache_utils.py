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
import glob
import os

from huggingface_hub import HfApi


def get_local_cached_files(cache_path, extension="*"):
    links = glob.glob(f"{cache_path}/**/*/*.{extension}", recursive=True)
    return [link for link in links if os.path.isfile(link)]


def check_traced_cache_entry(cache_path):
    local_files = get_local_cached_files(cache_path, "json")
    registry_path = [path for path in local_files if "REGISTRY" in path][0]
    registry_key = registry_path.split("/")[-1].replace(".json", "")
    local_files.remove(registry_path)
    local_hash_keys = []
    for local_file in local_files:
        local_hash_keys.append(local_file.split("/")[-2].replace("MODULE_", ""))
    assert registry_key in local_hash_keys


def assert_local_and_hub_cache_sync(cache_path, cache_repo_id):
    api = HfApi()
    remote_files = api.list_repo_files(cache_repo_id)
    local_files = get_local_cached_files(cache_path)
    for file in local_files:
        assert os.path.isfile(file)
        path_in_repo = file[len(cache_path) :].lstrip("/")
        assert path_in_repo in remote_files


def local_cache_size(cache_path):
    return len(get_local_cached_files(cache_path))
