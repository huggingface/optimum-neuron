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
"""HF Storage Bucket-based NEFF cache.

All neuron-specific cache logic lives here. Bucket I/O is delegated to a
long-running server process (bucket_server.py) that proxies the huggingface_hub
bucket API over a Unix socket.
"""

import atexit
import json
import os
import socket
import struct
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from ...utils import logging
from ..utils.cache_utils import get_neuron_cache_path
from ..utils.version_utils import get_neuronxcc_version
from ..version import __version__ as optimum_neuron_version
from .bucket_utils import (
    bucket_model_prefix,
    config_hash,
    get_cache_bucket,
    local_to_flat_bucket_path,
)


logger = logging.get_logger()

_BUCKET_SERVER_PATH = str(Path(__file__).parent / "bucket_server.py")
_UV_DEPS = ["huggingface_hub>=1.0"]

# Singleton server state
_server_process = None
_server_socket_path = None


# --- Server lifecycle ---


def _ensure_server() -> str:
    """Start the bucket server if not already running. Returns the socket path."""
    global _server_process, _server_socket_path

    if _server_process is not None and _server_process.poll() is None:
        return _server_socket_path

    try:
        import uv

        uv_bin = uv.find_uv_bin()
    except (ImportError, Exception):
        import shutil

        uv_bin = shutil.which("uv")
    if uv_bin is None:
        raise RuntimeError("uv is required for bucket cache operations. Install it: pip install uv")

    _server_socket_path = os.path.join(tempfile.gettempdir(), f"bucket_server_{os.getpid()}.sock")

    # Clean up stale socket from a previous crash
    if os.path.exists(_server_socket_path):
        os.unlink(_server_socket_path)

    cmd = [uv_bin, "run"]
    for dep in _UV_DEPS:
        cmd.extend(["--with", dep])
    cmd.extend(["python", _BUCKET_SERVER_PATH, _server_socket_path])

    _server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Wait for the server to be ready by polling the socket
    for _ in range(50):  # up to 5 seconds
        if _server_process.poll() is not None:
            stderr = _server_process.stderr.read().decode()
            pid = _server_process.pid
            _server_process = None
            raise RuntimeError(f"Bucket server (pid {pid}) exited early: {stderr[:500]}")
        if os.path.exists(_server_socket_path):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(_server_socket_path)
                sock.close()
                break
            except OSError:
                pass
        time.sleep(0.1)
    else:
        _server_process.kill()
        _server_process = None
        raise RuntimeError("Bucket server did not become ready within 5 seconds")

    atexit.register(_stop_server)
    return _server_socket_path


def _stop_server():
    global _server_process, _server_socket_path
    if _server_process is not None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
        _server_process = None
    if _server_socket_path and os.path.exists(_server_socket_path):
        os.unlink(_server_socket_path)
        _server_socket_path = None


def _recv_exact(sock, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


_TRANSIENT_ERROR_MARKERS = ("502", "503", "504", "429", "Timeout", "timeout", "EAGAIN")
_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # seconds, doubled each retry


def _call_server(method: str, **params) -> dict:
    """Send a request to the bucket server and return the response.

    Retries on transient HTTP errors (502, 503, 504, 429) with exponential backoff.
    """
    last_error = None
    backoff = _RETRY_BACKOFF

    for attempt in range(_MAX_RETRIES):
        socket_path = _ensure_server()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(socket_path)
            request = {"method": method, "params": params}
            payload = json.dumps(request).encode()
            if len(payload) > 2**32 - 1:
                raise ValueError(f"Request payload too large ({len(payload)} bytes)")
            sock.sendall(struct.pack(">I", len(payload)) + payload)

            raw_len = _recv_exact(sock, 4)
            if not raw_len:
                raise RuntimeError("Bucket server closed connection")
            msg_len = struct.unpack(">I", raw_len)[0]
            raw_msg = _recv_exact(sock, msg_len)
            result = json.loads(raw_msg.decode())
        finally:
            sock.close()

        if "error" not in result:
            return result

        error_msg = result["error"]
        if any(marker in error_msg for marker in _TRANSIENT_ERROR_MARKERS):
            last_error = error_msg
            logger.warning(f"Transient error on {method} (attempt {attempt + 1}/{_MAX_RETRIES}): {error_msg}")
            time.sleep(backoff)
            backoff *= 2
            continue

        raise RuntimeError(f"bucket_server {method}: {error_msg}")

    raise RuntimeError(f"bucket_server {method}: {last_error} (after {_MAX_RETRIES} retries)")


# --- Helpers ---


def _get_compiler_version(compiler_version: str | None = None) -> str:
    if compiler_version is not None:
        return compiler_version
    return get_neuronxcc_version()


def _get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    cache_url = os.environ.get("NEURON_COMPILE_CACHE_URL")
    if cache_url is not None:
        return Path(cache_url)
    path = get_neuron_cache_path()
    if path is None:
        raise ValueError("Neuron compile cache is disabled (--no-cache in NEURON_CC_FLAGS).")
    return path


def _list_local_modules(cache_dir: Path, compiler_version: str) -> set[str]:
    compiler_dir = cache_dir / f"neuronxcc-{compiler_version}"
    if not compiler_dir.exists():
        return set()
    return {d.name for d in compiler_dir.iterdir() if d.is_dir() and d.name.startswith("MODULE_")}


def _is_mounted(cache_dir: Path) -> bool:
    override = os.environ.get("NEURON_CACHE_MOUNTED")
    if override is not None:
        return override.lower() in ("1", "true", "yes")
    return os.path.ismount(cache_dir)


def _model_task_prefix(compiler_version: str, model_id: str, task: str | None) -> str:
    base = bucket_model_prefix(compiler_version, model_id)
    return f"{base}/{task or 'default'}"


# --- Public API ---


class CacheContext:
    """Mutable context yielded by hub_neuronx_cache.

    Allows callers to set ``export_config`` inside the ``with`` block
    (e.g. when the config is only known after compilation).
    """

    def __init__(self, export_config: dict | None = None):
        self.export_config = export_config


@contextmanager
def hub_neuronx_cache(
    model_id: str = "unknown",
    task: str | None = None,
    export_config: dict | None = None,
    compiler_version: str | None = None,
    bucket_id: str | None = None,
    cache_dir: str | Path | None = None,
):
    """Context manager for bucket-based NEFF cache.

    On enter: fetches cached NEFFs from bucket (if not mounted).
    On exit: uploads new NEFFs + export record to bucket.

    Yields a :class:`CacheContext` — set ``ctx.export_config`` inside the
    block when the config is only available after compilation.
    """

    compiler_version = _get_compiler_version(compiler_version)
    cache_dir = _get_cache_dir(cache_dir)
    bucket_id = bucket_id or get_cache_bucket()
    mounted = _is_mounted(cache_dir)

    if not mounted and bucket_id:
        try:
            fetch_cache(
                model_id, task=task, compiler_version=compiler_version, bucket_id=bucket_id, cache_dir=cache_dir
            )
        except Exception as e:
            logger.warning(f"Failed to fetch cache from bucket {bucket_id}: {e}")

    pre_modules = _list_local_modules(cache_dir, compiler_version)

    ctx = CacheContext(export_config)
    yield ctx

    post_modules = _list_local_modules(cache_dir, compiler_version)
    new_modules = post_modules - pre_modules

    if new_modules and bucket_id:
        try:
            sync_cache(
                model_id=model_id,
                task=task,
                export_config=ctx.export_config,
                new_modules=new_modules,
                compiler_version=compiler_version,
                bucket_id=bucket_id,
                cache_dir=cache_dir,
            )
        except Exception as e:
            logger.warning(f"Failed to sync cache to bucket {bucket_id}: {e}")


def fetch_cache(
    model_id: str,
    task: str | None = None,
    compiler_version: str | None = None,
    bucket_id: str | None = None,
    cache_dir: str | Path | None = None,
):
    """Pre-warm local cache by downloading MODULE dirs from the bucket."""
    compiler_version = _get_compiler_version(compiler_version)
    cache_dir = _get_cache_dir(cache_dir)
    bucket_id = bucket_id or get_cache_bucket()

    if not bucket_id:
        logger.warning("No cache bucket configured. Skipping fetch.")
        return

    model_prefix = _model_task_prefix(compiler_version, model_id, task)
    local_compiler_dir = cache_dir / f"neuronxcc-{compiler_version}"
    existing_modules = _list_local_modules(cache_dir, compiler_version)

    # List MODULE dirs in bucket
    result = _call_server("list_bucket_tree", bucket_id=bucket_id, prefix=model_prefix, recursive=False)
    module_folders = [
        item
        for item in result["items"]
        if item["type"] == "directory" and Path(item["path"]).name.startswith("MODULE_")
    ]

    # Collect files to download (skip existing modules)
    files_to_download = []
    for folder in module_folders:
        module_name = Path(folder["path"]).name
        if module_name in existing_modules:
            continue

        # List files in this MODULE dir
        files_result = _call_server("list_bucket_tree", bucket_id=bucket_id, prefix=folder["path"], recursive=True)
        for f in files_result["items"]:
            if f["type"] == "file":
                local_path = str(local_compiler_dir / module_name / Path(f["path"]).name)
                files_to_download.append([f["path"], local_path])

    if not files_to_download:
        logger.info(f"All MODULE dirs for {model_id} already in local cache.")
        return

    _call_server("download_bucket_files", bucket_id=bucket_id, files=files_to_download)
    logger.info(f"Downloaded {len(files_to_download)} files from bucket {bucket_id} for {model_id}.")


def sync_cache(
    model_id: str,
    task: str | None = None,
    export_config: dict | None = None,
    new_modules: set[str] | None = None,
    compiler_version: str | None = None,
    bucket_id: str | None = None,
    cache_dir: str | Path | None = None,
):
    """Upload new MODULE dirs and export record to the bucket."""
    compiler_version = _get_compiler_version(compiler_version)
    cache_dir = _get_cache_dir(cache_dir)
    bucket_id = bucket_id or get_cache_bucket()

    if not bucket_id or not new_modules:
        return

    local_compiler_dir = cache_dir / f"neuronxcc-{compiler_version}"
    model_task_prefix = _model_task_prefix(compiler_version, model_id, task)

    # Collect files for dual write (flat + per-model)
    files_to_add = []
    for module_name in sorted(new_modules):
        module_dir = local_compiler_dir / module_name
        if not module_dir.exists():
            continue
        for file_path in module_dir.iterdir():
            if not file_path.is_file():
                continue
            flat_path = f"{local_to_flat_bucket_path(module_name, compiler_version)}/{file_path.name}"
            files_to_add.append([str(file_path), flat_path])
            model_path = f"{model_task_prefix}/{module_name}/{file_path.name}"
            files_to_add.append([str(file_path), model_path])

    if files_to_add:
        _call_server("batch_bucket_files", bucket_id=bucket_id, add=files_to_add)
        logger.info(f"Uploaded {len(files_to_add)} files to bucket {bucket_id}.")

    # Upload export record last (signals completion)
    if export_config is not None:
        on_version = optimum_neuron_version
        hash_str = config_hash(export_config)
        record_path = f"{model_task_prefix}/exports/{on_version}/{hash_str}.json"
        content = json.dumps(export_config, sort_keys=True, indent=2)
        _call_server("batch_bucket_files", bucket_id=bucket_id, add=[[f"bytes:{content}", record_path]])
        logger.info(f"Uploaded export record: {record_path}")


def lookup_cache(
    model_id: str,
    task: str | None = None,
    compiler_version: str | None = None,
    on_version: str | None = None,
    bucket_id: str | None = None,
) -> list[dict]:
    """List previously exported configs for a model (advisory)."""
    compiler_version = _get_compiler_version(compiler_version)
    bucket_id = bucket_id or get_cache_bucket()

    if not bucket_id:
        return []

    if on_version is None:
        on_version = optimum_neuron_version

    model_task_prefix = _model_task_prefix(compiler_version, model_id, task)
    exports_prefix = f"{model_task_prefix}/exports/{on_version}"

    try:
        result = _call_server("list_bucket_tree", bucket_id=bucket_id, prefix=exports_prefix, recursive=True)
    except RuntimeError:
        return []

    json_files = [item for item in result["items"] if item["type"] == "file" and item["path"].endswith(".json")]
    if not json_files:
        return []

    # Download to temp dir and parse
    with tempfile.TemporaryDirectory() as tmpdir:
        files = [[f["path"], os.path.join(tmpdir, Path(f["path"]).name)] for f in json_files]
        _call_server("download_bucket_files", bucket_id=bucket_id, files=files)

        configs = []
        for _, local_path in files:
            try:
                with open(local_path) as fp:
                    configs.append(json.load(fp))
            except (json.JSONDecodeError, OSError):
                continue
        return configs
