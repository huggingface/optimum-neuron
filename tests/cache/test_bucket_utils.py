# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Tests for bucket_utils.py — bucket resolution, path helpers, config hashing."""

import os
import tempfile

from optimum.neuron.cache.bucket_utils import (
    bucket_flat_prefix,
    bucket_model_prefix,
    config_hash,
    get_cache_bucket,
    load_cache_bucket_from_hf_home,
    local_to_flat_bucket_path,
    set_cache_bucket_in_hf_home,
)


# --- config hash ---


def test_config_hash_deterministic():
    config = {"batch_size": 1, "sequence_length": 4096, "tp_degree": 8}
    assert config_hash(config) == config_hash(config)


def test_config_hash_length():
    assert len(config_hash({"batch_size": 1})) == 8


def test_config_hash_hex_chars():
    h = config_hash({"batch_size": 1})
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_order_independent():
    assert config_hash({"b": 2, "a": 1}) == config_hash({"a": 1, "b": 2})


def test_config_hash_different_configs():
    h1 = config_hash({"batch_size": 1, "sequence_length": 4096})
    h2 = config_hash({"batch_size": 4, "sequence_length": 2048})
    assert h1 != h2


def test_config_hash_nested():
    assert len(config_hash({"a": {"nested": True}, "b": [1, 2, 3]})) == 8


# --- bucket prefixes ---


def test_flat_prefix():
    assert bucket_flat_prefix("2.28.4405.0+abc123") == "neuronxcc-2.28.4405.0+abc123"


def test_model_prefix():
    assert (
        bucket_model_prefix("2.28.4405.0+abc123", "meta-llama/Llama-3.1-8B")
        == "neuronxcc-2.28.4405.0+abc123/meta-llama/Llama-3.1-8B"
    )


def test_model_prefix_no_slash():
    assert bucket_model_prefix("2.28.4405.0+abc123", "local-model") == "neuronxcc-2.28.4405.0+abc123/local-model"


# --- path translation ---


def test_local_to_flat():
    result = local_to_flat_bucket_path("MODULE_AAA+BBB", "2.28.4405.0+abc123")
    assert result == "neuronxcc-2.28.4405.0+abc123/MODULE_AAA+BBB"


def test_flat_matches_local_layout():
    """Flat bucket path should match the local cache directory structure."""
    compiler = "2.28.4405.0+abc123"
    module = "MODULE_X+Y"
    assert local_to_flat_bucket_path(module, compiler) == f"neuronxcc-{compiler}/{module}"


# --- bucket resolution ---


def test_default_bucket(monkeypatch):
    monkeypatch.delenv("NEURON_CACHE_BUCKET", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_file = os.path.join(tmpdir, "nonexistent")
        monkeypatch.setattr("optimum.neuron.cache.bucket_utils.HF_HOME_CACHE_BUCKET_FILE", fake_file)
        monkeypatch.setattr(
            "optimum.neuron.cache.bucket_utils.DEFAULT_CACHE_BUCKET", "aws-neuron/optimum-neuron-neff-cache"
        )
        assert get_cache_bucket() == "aws-neuron/optimum-neuron-neff-cache"


def test_env_var_overrides(monkeypatch):
    monkeypatch.setenv("NEURON_CACHE_BUCKET", "my-org/my-cache")
    assert get_cache_bucket() == "my-org/my-cache"


def test_local_file_overrides_default(monkeypatch):
    monkeypatch.delenv("NEURON_CACHE_BUCKET", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "bucket_file")
        with open(cache_file, "w") as f:
            f.write("custom-org/custom-cache")
        monkeypatch.setattr("optimum.neuron.cache.bucket_utils.HF_HOME_CACHE_BUCKET_FILE", cache_file)
        assert get_cache_bucket() == "custom-org/custom-cache"


def test_env_var_takes_priority_over_file(monkeypatch):
    monkeypatch.setenv("NEURON_CACHE_BUCKET", "env-bucket")
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "bucket_file")
        with open(cache_file, "w") as f:
            f.write("file-bucket")
        monkeypatch.setattr("optimum.neuron.cache.bucket_utils.HF_HOME_CACHE_BUCKET_FILE", cache_file)
        assert get_cache_bucket() == "env-bucket"


# --- set and load ---


def test_set_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        set_cache_bucket_in_hf_home("test-org/test-cache", hf_home=tmpdir)
        loaded = load_cache_bucket_from_hf_home(os.path.join(tmpdir, "optimum_neuron_cache_bucket"))
        assert loaded == "test-org/test-cache"


def test_load_nonexistent():
    assert load_cache_bucket_from_hf_home("/nonexistent/path/file") is None


def test_load_empty_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "empty")
        with open(filepath, "w") as f:
            f.write("")
        assert load_cache_bucket_from_hf_home(filepath) is None
