# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Unit tests for local Neuron compile cache cleanup utilities.

These tests use temporary directories and do not require Neuron hardware.
"""

from pathlib import Path
from unittest.mock import patch

from optimum.neuron.cache.cleanup import (
    CacheEntryState,
    CacheStatus,
    CleanupResult,
    _classify_entry,
    cleanup_local_cache,
    get_local_cache_status,
)


COMPILER_VERSION = "2.21.33363.0+82129205"
COMPILER_DIR = f"neuronxcc-{COMPILER_VERSION}"


def _make_success_entry(version_dir: Path, name: str = "MODULE_abc+def") -> Path:
    """Create a successful cache entry with neff and done marker."""
    entry = version_dir / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "model.hlo_module.pb").write_bytes(b"hlo data")
    (entry / "compile_flags.json").write_text('["--flag"]')
    (entry / "model.neff").write_bytes(b"x" * 1024)
    (entry / "model.done").write_bytes(b"")
    return entry


def _make_failed_entry(version_dir: Path, name: str = "MODULE_fail+xyz") -> Path:
    """Create a failed cache entry with a log file but no neff."""
    entry = version_dir / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "model.hlo_module.pb").write_bytes(b"hlo data")
    (entry / "compile_flags.json").write_text('["--flag"]')
    (entry / "model.log").write_text("ERROR: compilation failed")
    return entry


def _make_locked_entry(version_dir: Path, name: str = "MODULE_lock+abc") -> Path:
    """Create a locked cache entry with a lock file."""
    entry = version_dir / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "model.hlo_module.pb").write_bytes(b"hlo data")
    (entry / "model.hlo_module.pb.lock").write_bytes(b"")
    return entry


def _make_empty_entry(version_dir: Path, name: str = "MODULE_empty+abc") -> Path:
    """Create an empty cache entry (HLO only, no neff/log/lock)."""
    entry = version_dir / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "model.hlo_module.pb").write_bytes(b"hlo data")
    return entry


# --------------------------------------------------------------------------- #
# _classify_entry
# --------------------------------------------------------------------------- #


def test_classify_success(tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    entry = _make_success_entry(version_dir)
    assert _classify_entry(entry) == CacheEntryState.SUCCESS


def test_classify_failed(tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    entry = _make_failed_entry(version_dir)
    assert _classify_entry(entry) == CacheEntryState.FAILED


def test_classify_locked(tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    entry = _make_locked_entry(version_dir)
    assert _classify_entry(entry) == CacheEntryState.LOCKED


def test_classify_empty(tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    entry = _make_empty_entry(version_dir)
    assert _classify_entry(entry) == CacheEntryState.EMPTY


# --------------------------------------------------------------------------- #
# get_local_cache_status
# --------------------------------------------------------------------------- #


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_status_empty_cache(mock_version, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    status = get_local_cache_status(cache_dir)
    assert status.success_count == 0
    assert status.failed_count == 0
    assert status.total_size_bytes == 0
    assert status.compiler_versions == []


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_status_nonexistent_cache(mock_version, tmp_path):
    status = get_local_cache_status(tmp_path / "nonexistent")
    assert status.success_count == 0
    assert status.compiler_versions == []


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_status_counts_all_states(mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    _make_success_entry(version_dir, "MODULE_s1+a")
    _make_success_entry(version_dir, "MODULE_s2+b")
    _make_failed_entry(version_dir, "MODULE_f1+c")
    _make_locked_entry(version_dir, "MODULE_l1+d")
    _make_empty_entry(version_dir, "MODULE_e1+e")

    status = get_local_cache_status(tmp_path)
    assert status.success_count == 2
    assert status.failed_count == 1
    assert status.locked_count == 1
    assert status.empty_count == 1
    assert len(status.entries) == 5
    assert status.total_size_bytes > 0
    assert status.success_size_bytes > 0
    assert status.failed_size_bytes > 0


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_status_detects_current_compiler(mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    _make_success_entry(version_dir)

    status = get_local_cache_status(tmp_path)
    assert status.current_compiler_version == COMPILER_DIR
    assert COMPILER_DIR in status.compiler_versions


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_status_detects_old_versions(mock_version, tmp_path):
    # Current version
    current_dir = tmp_path / COMPILER_DIR
    _make_success_entry(current_dir, "MODULE_cur+a")
    # Old version
    old_dir = tmp_path / "neuronxcc-1.0.0.0+old"
    _make_success_entry(old_dir, "MODULE_old+b")

    status = get_local_cache_status(tmp_path)
    assert len(status.compiler_versions) == 2
    assert status.old_version_size_bytes > 0


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_status_ignores_non_module_dirs(mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    version_dir.mkdir(parents=True)
    # Registry directory should be ignored
    registry = version_dir / "0_REGISTRY"
    registry.mkdir()
    (registry / "some_file.json").write_text("{}")

    status = get_local_cache_status(tmp_path)
    assert status.success_count == 0
    assert len(status.entries) == 0


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_status_ignores_non_neuronxcc_dirs(mock_version, tmp_path):
    other_dir = tmp_path / "some_other_dir"
    other_dir.mkdir()
    (other_dir / "file.txt").write_text("data")

    status = get_local_cache_status(tmp_path)
    assert status.compiler_versions == []


# --------------------------------------------------------------------------- #
# CacheStatus.summary
# --------------------------------------------------------------------------- #


def test_status_summary_format():
    status = CacheStatus(
        cache_path="/var/tmp/neuron-compile-cache",
        compiler_versions=["neuronxcc-2.0"],
        current_compiler_version="neuronxcc-2.0",
        success_count=10,
        failed_count=2,
        total_size_bytes=1024 * 1024 * 50,
        success_size_bytes=1024 * 1024 * 48,
        failed_size_bytes=1024 * 1024 * 2,
    )
    summary = status.summary()
    assert "50.0 MB" in summary
    assert "Success:     10" in summary
    assert "Failed:       2" in summary


def test_status_summary_shows_old_version_info():
    status = CacheStatus(
        cache_path="/tmp/cache",
        old_version_size_bytes=1024 * 1024 * 100,
    )
    summary = status.summary()
    assert "Old compiler version data" in summary
    assert "100.0 MB" in summary


# --------------------------------------------------------------------------- #
# cleanup_local_cache — remove failed entries
# --------------------------------------------------------------------------- #


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_removes_failed_entries(mock_running, mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    success = _make_success_entry(version_dir, "MODULE_s+a")
    failed = _make_failed_entry(version_dir, "MODULE_f+b")

    result = cleanup_local_cache(cache_dir=tmp_path)
    assert result.failed_removed == 1
    assert result.bytes_freed > 0
    assert not failed.exists()
    assert success.exists()


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_dry_run_does_not_delete(mock_running, mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    failed = _make_failed_entry(version_dir)

    result = cleanup_local_cache(cache_dir=tmp_path, dry_run=True)
    assert result.failed_removed == 1
    assert result.bytes_freed > 0
    assert failed.exists()  # still there


# --------------------------------------------------------------------------- #
# cleanup_local_cache — remove stale locks
# --------------------------------------------------------------------------- #


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_removes_lock_files(mock_running, mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    locked = _make_locked_entry(version_dir)
    lock_file = locked / "model.hlo_module.pb.lock"
    assert lock_file.exists()

    result = cleanup_local_cache(cache_dir=tmp_path, remove_failed=False)
    assert result.locks_removed == 1
    assert not lock_file.exists()
    # The entry directory itself should still exist (only lock removed)
    assert locked.exists()


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=True)
def test_cleanup_exits_early_when_compiler_running(mock_running, mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    locked = _make_locked_entry(version_dir)
    failed = _make_failed_entry(version_dir, "MODULE_f+x")
    lock_file = locked / "model.hlo_module.pb.lock"

    result = cleanup_local_cache(cache_dir=tmp_path)
    # Entire cleanup is skipped when compiler is running
    assert result.locks_removed == 0
    assert result.failed_removed == 0
    assert result.skipped_locks_reason is not None
    assert "running" in result.skipped_locks_reason
    assert lock_file.exists()
    assert failed.exists()


# --------------------------------------------------------------------------- #
# cleanup_local_cache — remove empty entries
# --------------------------------------------------------------------------- #


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_skips_empty_by_default(mock_running, mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    empty = _make_empty_entry(version_dir)

    result = cleanup_local_cache(cache_dir=tmp_path)
    assert result.empty_removed == 0
    assert empty.exists()


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_removes_empty_when_requested(mock_running, mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    empty = _make_empty_entry(version_dir)

    result = cleanup_local_cache(cache_dir=tmp_path, remove_empty=True)
    assert result.empty_removed == 1
    assert not empty.exists()


# --------------------------------------------------------------------------- #
# cleanup_local_cache — remove old versions
# --------------------------------------------------------------------------- #


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_removes_old_versions(mock_running, mock_version, tmp_path):
    # Current version
    current_dir = tmp_path / COMPILER_DIR
    _make_success_entry(current_dir, "MODULE_cur+a")
    # Old version
    old_dir = tmp_path / "neuronxcc-1.0.0.0+old"
    _make_success_entry(old_dir, "MODULE_old+b")

    result = cleanup_local_cache(cache_dir=tmp_path, remove_old_versions=True)
    assert "neuronxcc-1.0.0.0+old" in result.old_versions_removed
    assert result.bytes_freed > 0
    assert not old_dir.exists()
    assert current_dir.exists()


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_keeps_old_versions_by_default(mock_running, mock_version, tmp_path):
    old_dir = tmp_path / "neuronxcc-1.0.0.0+old"
    _make_success_entry(old_dir, "MODULE_old+a")

    result = cleanup_local_cache(cache_dir=tmp_path)
    assert result.old_versions_removed == []
    assert old_dir.exists()


# --------------------------------------------------------------------------- #
# cleanup_local_cache — wipe
# --------------------------------------------------------------------------- #


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_cleanup_wipe(mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    _make_success_entry(version_dir, "MODULE_s+a")
    _make_failed_entry(version_dir, "MODULE_f+b")

    result = cleanup_local_cache(cache_dir=tmp_path, wipe=True)
    assert result.wiped
    assert result.bytes_freed > 0
    assert not tmp_path.exists()


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
def test_cleanup_wipe_dry_run(mock_version, tmp_path):
    version_dir = tmp_path / COMPILER_DIR
    _make_success_entry(version_dir)

    result = cleanup_local_cache(cache_dir=tmp_path, wipe=True, dry_run=True)
    assert result.wiped
    assert result.bytes_freed > 0
    assert tmp_path.exists()  # still there


# --------------------------------------------------------------------------- #
# cleanup_local_cache — edge cases
# --------------------------------------------------------------------------- #


def test_cleanup_nonexistent_cache(tmp_path):
    result = cleanup_local_cache(cache_dir=tmp_path / "nonexistent")
    assert result.failed_removed == 0
    assert result.locks_removed == 0
    assert result.bytes_freed == 0


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=COMPILER_VERSION)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_mixed_entries(mock_running, mock_version, tmp_path):
    """Test cleanup with all entry types in one pass."""
    version_dir = tmp_path / COMPILER_DIR
    success = _make_success_entry(version_dir, "MODULE_s+a")
    failed = _make_failed_entry(version_dir, "MODULE_f+b")
    locked = _make_locked_entry(version_dir, "MODULE_l+c")
    empty = _make_empty_entry(version_dir, "MODULE_e+d")

    result = cleanup_local_cache(cache_dir=tmp_path, remove_empty=True)
    assert result.failed_removed == 1
    assert result.locks_removed == 1
    assert result.empty_removed == 1
    assert not failed.exists()
    assert not (locked / "model.hlo_module.pb.lock").exists()
    assert not empty.exists()
    assert success.exists()


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=None)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_works_without_compiler_installed(mock_running, mock_version, tmp_path):
    """Cleanup should work even if neuronx-cc is not installed (version=None)."""
    version_dir = tmp_path / COMPILER_DIR
    _make_failed_entry(version_dir)

    result = cleanup_local_cache(cache_dir=tmp_path)
    assert result.failed_removed == 1


@patch("optimum.neuron.cache.cleanup._get_current_compiler_version", return_value=None)
@patch("optimum.neuron.cache.cleanup._is_neuronx_cc_running", return_value=False)
def test_cleanup_no_old_version_removal_without_compiler(mock_running, mock_version, tmp_path):
    """Without a known compiler version, no version can be classified as old."""
    version_dir = tmp_path / "neuronxcc-1.0.0.0+old"
    _make_success_entry(version_dir)

    result = cleanup_local_cache(cache_dir=tmp_path, remove_old_versions=True)
    # Can't determine what's old without knowing the current version
    assert result.old_versions_removed == []
    assert version_dir.exists()


# --------------------------------------------------------------------------- #
# CleanupResult.summary
# --------------------------------------------------------------------------- #


def test_cleanup_result_summary_nothing():
    result = CleanupResult()
    assert "Nothing to clean up" in result.summary()


def test_cleanup_result_summary_mixed():
    result = CleanupResult(
        failed_removed=3,
        locks_removed=1,
        bytes_freed=2048,
    )
    summary = result.summary()
    assert "3 failed" in summary
    assert "1 stale lock" in summary
    assert "2.0 KB" in summary


def test_cleanup_result_summary_wipe():
    result = CleanupResult(wiped=True, bytes_freed=1024 * 1024 * 500)
    summary = result.summary()
    assert "wiped" in summary.lower()
    assert "500.0 MB" in summary


def test_cleanup_result_summary_skipped_locks():
    result = CleanupResult(skipped_locks_reason="neuronx-cc processes are currently running")
    summary = result.summary()
    assert "Skipped lock cleanup" in summary


# --------------------------------------------------------------------------- #
# CacheStatus._format_size
# --------------------------------------------------------------------------- #


def test_format_size_bytes():
    assert CacheStatus._format_size(500) == "500 B"


def test_format_size_kb():
    assert CacheStatus._format_size(2048) == "2.0 KB"


def test_format_size_mb():
    assert CacheStatus._format_size(5 * 1024 * 1024) == "5.0 MB"


def test_format_size_gb():
    assert CacheStatus._format_size(3 * 1024 * 1024 * 1024) == "3.0 GB"
