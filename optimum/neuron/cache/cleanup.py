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
"""Local Neuron compile cache inspection and cleanup utilities."""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)

# Default local compile cache path
_DEFAULT_CACHE_PATH = "/var/tmp/neuron-compile-cache"


class CacheEntryState(str, Enum):
    """Possible states of a compile cache entry directory."""

    SUCCESS = "success"
    FAILED = "failed"
    LOCKED = "locked"
    EMPTY = "empty"


@dataclass
class CacheEntryStatus:
    """Status of a single compile cache entry directory."""

    path: str
    state: CacheEntryState
    size_bytes: int = 0


@dataclass
class CacheStatus:
    """Aggregated status of the local Neuron compile cache."""

    cache_path: str
    compiler_versions: list[str] = field(default_factory=list)
    current_compiler_version: str | None = None
    success_count: int = 0
    failed_count: int = 0
    locked_count: int = 0
    empty_count: int = 0
    total_size_bytes: int = 0
    success_size_bytes: int = 0
    failed_size_bytes: int = 0
    locked_size_bytes: int = 0
    empty_size_bytes: int = 0
    old_version_size_bytes: int = 0
    entries: list[CacheEntryStatus] = field(default_factory=list)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def summary(self) -> str:
        lines = [
            f"Cache path: {self.cache_path}",
            f"Total size: {self._format_size(self.total_size_bytes)}",
            "",
            "Compiler versions:",
        ]
        for v in self.compiler_versions:
            current = (
                " (current)"
                if self.current_compiler_version and v.startswith(self.current_compiler_version)
                else " (old)"
            )
            lines.append(f"  - {v}{current}")
        lines.append("")
        lines.append("Entry counts:")
        lines.append(f"  Success: {self.success_count:>6}  ({self._format_size(self.success_size_bytes)})")
        lines.append(f"  Failed:  {self.failed_count:>6}  ({self._format_size(self.failed_size_bytes)})")
        lines.append(f"  Locked:  {self.locked_count:>6}  ({self._format_size(self.locked_size_bytes)})")
        lines.append(f"  Empty:   {self.empty_count:>6}  ({self._format_size(self.empty_size_bytes)})")
        if self.old_version_size_bytes > 0:
            lines.append(f"\nOld compiler version data: {self._format_size(self.old_version_size_bytes)}")
        return "\n".join(lines)


@dataclass
class CleanupResult:
    """Result of a cache cleanup operation."""

    failed_removed: int = 0
    locks_removed: int = 0
    empty_removed: int = 0
    old_versions_removed: list[str] = field(default_factory=list)
    wiped: bool = False
    bytes_freed: int = 0
    skipped_locks_reason: str | None = None

    def summary(self) -> str:
        if self.wiped:
            return f"Cache wiped. Freed {CacheStatus._format_size(self.bytes_freed)}."
        parts = []
        if self.failed_removed > 0:
            parts.append(f"Removed {self.failed_removed} failed entries")
        if self.locks_removed > 0:
            parts.append(f"Removed {self.locks_removed} stale lock files")
        if self.skipped_locks_reason:
            parts.append(f"Skipped lock cleanup: {self.skipped_locks_reason}")
        if self.empty_removed > 0:
            parts.append(f"Removed {self.empty_removed} empty entries")
        if self.old_versions_removed:
            parts.append(f"Removed old compiler versions: {', '.join(self.old_versions_removed)}")
        if not parts:
            parts.append("Nothing to clean up")
        parts.append(f"Freed {CacheStatus._format_size(self.bytes_freed)}")
        return ". ".join(parts) + "."


def _get_cache_path(cache_dir: str | Path | None = None) -> Path:
    """Resolve the local compile cache path."""
    if cache_dir is not None:
        return Path(cache_dir)
    env_url = os.environ.get("NEURON_COMPILE_CACHE_URL")
    if env_url and not env_url.startswith("s3://"):
        return Path(env_url)
    return Path(_DEFAULT_CACHE_PATH)


def _get_current_compiler_version() -> str | None:
    """Get the current neuronx-cc compiler version, or None if unavailable."""
    try:
        from ..utils.version_utils import get_neuronxcc_version

        return get_neuronxcc_version()
    except Exception:
        return None


def _dir_size(path: Path) -> int:
    """Compute the total size of a directory in bytes."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
    except OSError:
        pass
    return total


def _classify_entry(entry_dir: Path) -> CacheEntryState:
    """Classify a MODULE_* entry directory into a state."""
    has_neff = (entry_dir / "model.neff").exists()
    has_done = (entry_dir / "model.done").exists()
    has_log = (entry_dir / "model.log").exists()
    has_lock = any(entry_dir.glob("*.lock"))
    if has_neff and has_done:
        return CacheEntryState.SUCCESS
    if has_log and not has_neff:
        return CacheEntryState.FAILED
    if has_lock:
        return CacheEntryState.LOCKED
    return CacheEntryState.EMPTY


def get_local_cache_status(cache_dir: str | Path | None = None) -> CacheStatus:
    """Get the status of the local Neuron compile cache.

    Args:
        cache_dir: Override the cache directory path. If None, uses the environment
            variable NEURON_COMPILE_CACHE_URL or the default /var/tmp/neuron-compile-cache.

    Returns:
        A CacheStatus dataclass with counts, sizes, and per-entry details.
    """
    cache_path = _get_cache_path(cache_dir)
    status = CacheStatus(cache_path=str(cache_path))

    if not cache_path.is_dir():
        return status

    current_version = _get_current_compiler_version()
    current_prefix = f"neuronxcc-{current_version}" if current_version else None
    status.current_compiler_version = current_prefix

    for version_dir in sorted(cache_path.iterdir()):
        if not version_dir.is_dir() or not version_dir.name.startswith("neuronxcc-"):
            continue
        status.compiler_versions.append(version_dir.name)
        is_old = current_prefix is not None and not version_dir.name.startswith(current_prefix)

        for entry_dir in version_dir.iterdir():
            if not entry_dir.is_dir() or not entry_dir.name.startswith("MODULE_"):
                continue
            state = _classify_entry(entry_dir)
            size = _dir_size(entry_dir)
            status.entries.append(CacheEntryStatus(path=str(entry_dir), state=state, size_bytes=size))
            status.total_size_bytes += size

            if state == CacheEntryState.SUCCESS:
                status.success_count += 1
                status.success_size_bytes += size
            elif state == CacheEntryState.FAILED:
                status.failed_count += 1
                status.failed_size_bytes += size
            elif state == CacheEntryState.LOCKED:
                status.locked_count += 1
                status.locked_size_bytes += size
            else:
                status.empty_count += 1
                status.empty_size_bytes += size

            if is_old:
                status.old_version_size_bytes += size

    return status


def _is_neuronx_cc_running() -> bool:
    """Check if any neuronx-cc compiler processes are currently running."""
    try:
        result = subprocess.run(["pgrep", "-c", "neuronx-cc"], capture_output=True, text=True)
        count = int(result.stdout.strip() or "0")
        return count > 0
    except (OSError, ValueError):
        return False


def cleanup_local_cache(
    cache_dir: str | Path | None = None,
    remove_failed: bool = True,
    remove_locks: bool = True,
    remove_empty: bool = False,
    remove_old_versions: bool = False,
    wipe: bool = False,
    dry_run: bool = False,
) -> CleanupResult:
    """Clean up the local Neuron compile cache.

    Removes poisoned entries (failed compilations, stale locks) and optionally
    empty entries or old compiler version caches.

    Args:
        cache_dir: Override the cache directory path.
        remove_failed: Remove entries with cached compilation failures (model.log without model.neff).
        remove_locks: Remove stale lock files (only if no neuronx-cc processes are running).
        remove_empty: Remove empty/incomplete entries (no neff, no log, no lock).
        remove_old_versions: Remove entire cache directories for old compiler versions.
        wipe: Remove the entire cache directory.
        dry_run: If True, report what would be cleaned without actually removing anything.

    Returns:
        A CleanupResult with counts and bytes freed.
    """
    cache_path = _get_cache_path(cache_dir)
    result = CleanupResult()

    if not cache_path.is_dir():
        return result

    if wipe:
        result.bytes_freed = _dir_size(cache_path)
        result.wiped = True
        if not dry_run:
            shutil.rmtree(cache_path)
        return result

    current_version = _get_current_compiler_version()
    current_prefix = f"neuronxcc-{current_version}" if current_version else None

    # Check for active compiler processes before lock cleanup
    compiler_running = False
    if remove_locks:
        compiler_running = _is_neuronx_cc_running()
        if compiler_running:
            result.skipped_locks_reason = "neuronx-cc processes are currently running"

    for version_dir in sorted(cache_path.iterdir()):
        if not version_dir.is_dir() or not version_dir.name.startswith("neuronxcc-"):
            continue

        is_old = current_prefix is not None and not version_dir.name.startswith(current_prefix)

        # Remove entire old version directories
        if remove_old_versions and is_old:
            size = _dir_size(version_dir)
            result.bytes_freed += size
            result.old_versions_removed.append(version_dir.name)
            if not dry_run:
                shutil.rmtree(version_dir)
            continue

        for entry_dir in list(version_dir.iterdir()):
            if not entry_dir.is_dir() or not entry_dir.name.startswith("MODULE_"):
                continue
            state = _classify_entry(entry_dir)

            if state == CacheEntryState.FAILED and remove_failed:
                size = _dir_size(entry_dir)
                result.bytes_freed += size
                result.failed_removed += 1
                if not dry_run:
                    shutil.rmtree(entry_dir)

            elif state == CacheEntryState.LOCKED and remove_locks and not compiler_running:
                # Only remove lock files, not the entire entry
                for lock_file in entry_dir.glob("*.lock"):
                    size = lock_file.stat().st_size if lock_file.is_file() else 0
                    result.bytes_freed += size
                    result.locks_removed += 1
                    if not dry_run:
                        lock_file.unlink()

            elif state == CacheEntryState.EMPTY and remove_empty:
                size = _dir_size(entry_dir)
                result.bytes_freed += size
                result.empty_removed += 1
                if not dry_run:
                    shutil.rmtree(entry_dir)

    return result
