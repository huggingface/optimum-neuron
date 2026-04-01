#!/usr/bin/env python
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
"""HF Storage Bucket proxy server.

A generic proxy mirroring the huggingface_hub bucket API over a Unix socket.
Runs in an isolated environment (via uv) with huggingface_hub >= 1.0.
Contains no neuron-specific logic — all cache semantics live in bucket_cache.py.

Protocol:
    Request:  4-byte big-endian length prefix + UTF-8 JSON payload
    Response: 4-byte big-endian length prefix + UTF-8 JSON payload

Request format:  {"method": "<name>", "params": {<kwargs>}}
Response format: {<result dict>} or {"error": "<message>"}
"""

import json
import os
import signal
import socketserver
import struct
import sys
from pathlib import Path


# Suppress progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import HfApi  # noqa: E402


def _recv_exact(sock, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _recv_message(sock) -> dict | None:
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    raw_msg = _recv_exact(sock, msg_len)
    return json.loads(raw_msg.decode())


_MAX_PAYLOAD = 2**32 - 1


def _send_message(sock, data: dict):
    payload = json.dumps(data).encode()
    if len(payload) > _MAX_PAYLOAD:
        raise ValueError(f"Response payload too large ({len(payload)} bytes, max {_MAX_PAYLOAD})")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


# --- API methods (generic huggingface_hub proxy) ---


def list_bucket_tree(bucket_id: str, prefix: str | None = None, recursive: bool = False) -> dict:
    """Proxy for HfApi.list_bucket_tree(). Returns serializable items."""
    api = HfApi()
    items = list(api.list_bucket_tree(bucket_id, prefix=prefix, recursive=recursive))
    result = []
    for item in items:
        entry = {"path": item.path, "type": getattr(item, "type", "unknown")}
        if hasattr(item, "size") and item.size is not None:
            entry["size"] = item.size
        result.append(entry)
    return {"items": result}


def download_bucket_files(bucket_id: str, files: list[list[str]]) -> dict:
    """Proxy for HfApi.download_bucket_files().

    Args:
        files: list of [remote_path, local_path] pairs.
    """
    api = HfApi()
    file_tuples = [(remote, local) for remote, local in files]
    # Ensure local directories exist
    for _, local_path in file_tuples:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    api.download_bucket_files(bucket_id, files=file_tuples)
    return {"downloaded": len(file_tuples)}


def batch_bucket_files(
    bucket_id: str,
    add: list[list[str]] | None = None,
    delete: list[str] | None = None,
) -> dict:
    """Proxy for HfApi.batch_bucket_files().

    Args:
        add: list of [source, destination] pairs. Source is a local file path or
             a string to be encoded as bytes (prefixed with "bytes:").
        delete: list of remote paths to delete.
    """
    api = HfApi()
    add_ops = None
    if add:
        add_ops = []
        for source, dest in add:
            if source.startswith("bytes:"):
                add_ops.append((source[6:].encode(), dest))
            else:
                add_ops.append((source, dest))
    delete_ops = delete if delete else None
    api.batch_bucket_files(bucket_id, add=add_ops, delete=delete_ops)
    return {"added": len(add_ops) if add_ops else 0, "deleted": len(delete_ops) if delete_ops else 0}


def ping() -> dict:
    return {"status": "ok"}


METHODS = {
    "list_bucket_tree": list_bucket_tree,
    "download_bucket_files": download_bucket_files,
    "batch_bucket_files": batch_bucket_files,
    "ping": ping,
}


class BucketHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            request = _recv_message(self.request)
            if request is None:
                break
            try:
                method = request.get("method")
                params = request.get("params", {})
                handler = METHODS.get(method)
                if handler is None:
                    result = {"error": f"Unknown method: {method}"}
                else:
                    result = handler(**params)
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
            _send_message(self.request, result)


class BucketServer(socketserver.UnixStreamServer):
    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        super().__init__(socket_path, BucketHandler)

    def server_close(self):
        super().server_close()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: bucket_server.py <socket_path>", file=sys.stderr)
        sys.exit(1)

    socket_path = sys.argv[1]
    server = BucketServer(socket_path)

    def shutdown(signum, frame):
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    # Signal readiness
    print(os.getpid(), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
