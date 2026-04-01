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
"""Tests for bucket_cache.py — context manager, fetch, sync, lookup via bucket server."""

import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from optimum.neuron.cache.bucket_cache import (
    _call_server,
    _list_local_modules,
    _stop_server,
    fetch_cache,
    hub_neuronx_cache,
    lookup_cache,
    sync_cache,
)
from optimum.neuron.cache.bucket_utils import bucket_model_prefix


TEST_BUCKET = "optimum-internal-testing/optimum-neuron-neff-cache"
TEST_MODEL_ID = "test-org/test-model"
TEST_TASK = "feature-extraction"
TEST_EXPORT_CONFIG = {
    "batch_size": 1,
    "sequence_length": 4096,
    "tp_degree": 8,
    "torch_dtype": "bfloat16",
    "on_device_sampling": True,
}


@pytest.fixture(autouse=True)
def _check_uv():
    if shutil.which("uv") is None:
        pytest.skip("uv is required for bucket cache tests")


@pytest.fixture()
def compiler_version():
    """Unique compiler version per test for isolation."""
    version = f"0.0.0+test{uuid.uuid4().hex[:8]}"
    yield version
    prefix = f"neuronxcc-{version}"
    try:
        result = _call_server("list_bucket_tree", bucket_id=TEST_BUCKET, prefix=prefix, recursive=True)
        files = [item["path"] for item in result["items"] if item["type"] == "file"]
        if files:
            _call_server("batch_bucket_files", bucket_id=TEST_BUCKET, delete=files)
    except Exception:
        pass


@pytest.fixture(autouse=True, scope="session")
def _shutdown_server():
    """Stop the bucket server after all tests."""
    yield
    _stop_server()


def _list_bucket_modules(bucket_id, prefix):
    """List MODULE dir names under a bucket prefix using the generic server API."""
    result = _call_server("list_bucket_tree", bucket_id=bucket_id, prefix=prefix, recursive=False)
    return [
        Path(item["path"]).name
        for item in result["items"]
        if item["type"] == "directory" and Path(item["path"]).name.startswith("MODULE_")
    ]


def _create_fake_module(cache_dir, compiler_version, module_name, files=None):
    module_dir = cache_dir / f"neuronxcc-{compiler_version}" / module_name
    module_dir.mkdir(parents=True, exist_ok=True)
    if files is None:
        files = {"model.neff": b"fake neff content", "model.done": b""}
    for name, content in files.items():
        (module_dir / name).write_bytes(content)
    return module_dir


# --- server ---


def test_server_ping():
    result = _call_server("ping")
    assert result["status"] == "ok"


def test_server_list_modules_empty():
    assert _list_bucket_modules(TEST_BUCKET, "neuronxcc-0.0.0+nonexistent/nonexistent") == []


def test_server_unknown_method():
    with pytest.raises(RuntimeError, match="Unknown method"):
        _call_server("nonexistent_method")


# --- list local modules ---


def test_list_local_modules_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert _list_local_modules(Path(tmpdir), "1.0.0+abc") == set()


def test_list_local_modules_finds_modules():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        _create_fake_module(cache_dir, "1.0.0+abc", "MODULE_AAA+BBB")
        _create_fake_module(cache_dir, "1.0.0+abc", "MODULE_CCC+DDD")
        (cache_dir / "neuronxcc-1.0.0+abc" / "some_other_dir").mkdir()

        result = _list_local_modules(cache_dir, "1.0.0+abc")
        assert result == {"MODULE_AAA+BBB", "MODULE_CCC+DDD"}


# --- sync ---


def test_sync_uploads_to_both_areas(compiler_version):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        _create_fake_module(cache_dir, compiler_version, "MODULE_TEST1+HASH1")

        sync_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            export_config=TEST_EXPORT_CONFIG,
            new_modules={"MODULE_TEST1+HASH1"},
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=cache_dir,
        )

        # Flat area
        flat_modules = _list_bucket_modules(TEST_BUCKET, f"neuronxcc-{compiler_version}")
        assert "MODULE_TEST1+HASH1" in flat_modules

        # Per-model+task area
        model_prefix = f"{bucket_model_prefix(compiler_version, TEST_MODEL_ID)}/{TEST_TASK}"
        model_modules = _list_bucket_modules(TEST_BUCKET, model_prefix)
        assert "MODULE_TEST1+HASH1" in model_modules


def test_sync_without_export_config(compiler_version):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        _create_fake_module(cache_dir, compiler_version, "MODULE_TEST2+HASH2")

        sync_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            export_config=None,
            new_modules={"MODULE_TEST2+HASH2"},
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=cache_dir,
        )

        configs = lookup_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
        )
        assert configs == []


# --- fetch ---


def test_fetch_downloads_modules(compiler_version):
    with tempfile.TemporaryDirectory() as tmpdir_upload, tempfile.TemporaryDirectory() as tmpdir_download:
        upload_dir = Path(tmpdir_upload)
        download_dir = Path(tmpdir_download)

        _create_fake_module(upload_dir, compiler_version, "MODULE_FETCH1+HASH1")
        sync_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            export_config=None,
            new_modules={"MODULE_FETCH1+HASH1"},
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=upload_dir,
        )

        fetch_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=download_dir,
        )

        modules = _list_local_modules(download_dir, compiler_version)
        assert "MODULE_FETCH1+HASH1" in modules

        neff_path = download_dir / f"neuronxcc-{compiler_version}" / "MODULE_FETCH1+HASH1" / "model.neff"
        assert neff_path.exists()


def test_fetch_skips_existing(compiler_version):
    with tempfile.TemporaryDirectory() as tmpdir_upload, tempfile.TemporaryDirectory() as tmpdir_download:
        upload_dir = Path(tmpdir_upload)
        download_dir = Path(tmpdir_download)

        _create_fake_module(upload_dir, compiler_version, "MODULE_SKIP1+HASH1")
        sync_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            export_config=None,
            new_modules={"MODULE_SKIP1+HASH1"},
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=upload_dir,
        )

        _create_fake_module(download_dir, compiler_version, "MODULE_SKIP1+HASH1", {"model.neff": b"local content"})

        fetch_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=download_dir,
        )

        neff_path = download_dir / f"neuronxcc-{compiler_version}" / "MODULE_SKIP1+HASH1" / "model.neff"
        assert neff_path.read_bytes() == b"local content"


# --- lookup ---


def test_lookup_returns_configs(compiler_version):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        _create_fake_module(cache_dir, compiler_version, "MODULE_LOOK1+HASH1")

        sync_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            export_config=TEST_EXPORT_CONFIG,
            new_modules={"MODULE_LOOK1+HASH1"},
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=cache_dir,
        )

        from optimum.neuron.version import __version__

        configs = lookup_cache(
            model_id=TEST_MODEL_ID,
            task=TEST_TASK,
            compiler_version=compiler_version,
            on_version=__version__,
            bucket_id=TEST_BUCKET,
        )

        assert len(configs) == 1
        assert configs[0]["batch_size"] == 1
        assert configs[0]["tp_degree"] == 8


def test_lookup_empty(compiler_version):
    configs = lookup_cache(
        model_id="nonexistent/model",
        compiler_version=compiler_version,
        bucket_id=TEST_BUCKET,
    )
    assert configs == []


# --- context manager ---


def test_context_no_new_modules(compiler_version):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        with hub_neuronx_cache(
            TEST_MODEL_ID,
            task=TEST_TASK,
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=cache_dir,
        ):
            pass

        assert _list_bucket_modules(TEST_BUCKET, f"neuronxcc-{compiler_version}") == []


def test_context_detects_new_modules(compiler_version):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        with hub_neuronx_cache(
            TEST_MODEL_ID,
            task=TEST_TASK,
            export_config=TEST_EXPORT_CONFIG,
            compiler_version=compiler_version,
            bucket_id=TEST_BUCKET,
            cache_dir=cache_dir,
        ):
            _create_fake_module(cache_dir, compiler_version, "MODULE_CTX1+HASH1")

        module_names = _list_bucket_modules(TEST_BUCKET, f"neuronxcc-{compiler_version}")
        assert "MODULE_CTX1+HASH1" in module_names
