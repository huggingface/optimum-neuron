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
"""Manage multiple independent vLLM server subprocesses for data-parallel serving."""

import logging
import os
import signal
import subprocess
import sys
import time


logger = logging.getLogger("ServerManager")


class VLLMServerManager:
    """Spawn and manage N independent vLLM server processes, each on its own NeuronCores."""

    def __init__(
        self,
        server_count: int,
        base_port: int,
        tp_size: int,
        vllm_args: list[str],
    ):
        self.server_count = server_count
        self.base_port = base_port
        self.tp_size = tp_size
        self.vllm_args = vllm_args
        self._processes: list[subprocess.Popen] = []

    def _core_range(self, rank: int) -> str:
        """Compute the NEURON_RT_VISIBLE_CORES value for a given DP rank."""
        start = rank * self.tp_size
        end = start + self.tp_size - 1
        if start == end:
            return str(start)
        return f"{start}-{end}"

    def start(self):
        """Launch all vLLM server subprocesses."""
        for rank in range(self.server_count):
            port = self.base_port + 1 + rank
            core_range = self._core_range(rank)

            env = os.environ.copy()
            env["NEURON_RT_VISIBLE_CORES"] = core_range
            env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            cmd = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--port",
                str(port),
                "--host",
                "127.0.0.1",
            ] + self.vllm_args

            logger.info("Starting vLLM server rank=%d port=%d cores=%s", rank, port, core_range)
            proc = subprocess.Popen(cmd, env=env)
            self._processes.append(proc)

    def monitor(self) -> int:
        """Wait for any subprocess to exit. Returns the exit code of the first to finish."""
        while True:
            for proc in self._processes:
                ret = proc.poll()
                if ret is not None:
                    logger.warning("vLLM server pid=%d exited with code %d", proc.pid, ret)
                    return ret
            time.sleep(1)

    def shutdown(self):
        """Gracefully stop all subprocesses."""
        for proc in self._processes:
            if proc.poll() is None:
                logger.info("Sending SIGTERM to vLLM server pid=%d", proc.pid)
                proc.send_signal(signal.SIGTERM)

        # Wait for graceful shutdown.
        deadline = time.monotonic() + 10
        for proc in self._processes:
            remaining = max(0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        # Force kill any remaining.
        for proc in self._processes:
            if proc.poll() is None:
                logger.warning("Force killing vLLM server pid=%d", proc.pid)
                proc.kill()
                proc.wait()

        self._processes.clear()
