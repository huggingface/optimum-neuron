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
import subprocess
import sys
import threading
import time


logger = logging.getLogger("ServerManager")


class VLLMServerManager:
    """Spawn and manage N independent vLLM server processes, each on its own NeuronCores."""

    def __init__(
        self,
        ports: list[int],
        tp_size: int,
        vllm_args: list[str],
    ):
        self.ports = ports
        self.server_count = len(ports)
        self.tp_size = tp_size
        self.vllm_args = vllm_args
        self._processes: list[subprocess.Popen] = []
        self._stop = threading.Event()

    @staticmethod
    def _resolve_physical_cores() -> list[int] | None:
        """Parse NEURON_RT_VISIBLE_CORES into a list of physical core IDs.

        Returns None when the variable is unset (meaning all cores are available
        starting from 0).
        """
        env_var = os.environ.get("NEURON_RT_VISIBLE_CORES")
        if env_var is None:
            return None
        env_var = env_var.strip()
        if "," in env_var:
            return [int(c.strip()) for c in env_var.split(",")]
        if "-" in env_var:
            start, end = env_var.split("-")
            return list(range(int(start.strip()), int(end.strip()) + 1))
        return [int(env_var)]

    def _core_range(self, rank: int) -> str:
        """Compute the NEURON_RT_VISIBLE_CORES value for a given DP rank."""
        physical_cores = self._resolve_physical_cores()
        if physical_cores is not None:
            # Partition the pre-existing visible cores across DP ranks.
            cores_for_rank = physical_cores[rank * self.tp_size : (rank + 1) * self.tp_size]
            if len(cores_for_rank) < self.tp_size:
                raise ValueError(
                    f"DP rank {rank} needs {self.tp_size} cores but only "
                    f"{len(cores_for_rank)} are available from NEURON_RT_VISIBLE_CORES."
                )
            if cores_for_rank == list(range(cores_for_rank[0], cores_for_rank[-1] + 1)):
                # Contiguous range — use compact notation.
                if len(cores_for_rank) == 1:
                    return str(cores_for_rank[0])
                return f"{cores_for_rank[0]}-{cores_for_rank[-1]}"
            return ",".join(str(c) for c in cores_for_rank)
        # No constraint: assign from core 0.
        start = rank * self.tp_size
        end = start + self.tp_size - 1
        if start == end:
            return str(start)
        return f"{start}-{end}"

    def start(self):
        """Launch all vLLM server subprocesses."""
        for rank, port in enumerate(self.ports):
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

    def monitor(self) -> int | None:
        """Wait for any subprocess to exit. Returns the exit code of the first to finish.

        Returns None if stopped via shutdown().
        """
        while not self._stop.is_set():
            for proc in self._processes:
                ret = proc.poll()
                if ret is not None:
                    logger.warning("vLLM server pid=%d exited with code %d", proc.pid, ret)
                    return ret
            self._stop.wait(1)
        return None

    def shutdown(self):
        """Gracefully stop all subprocesses."""
        self._stop.set()
        for proc in self._processes:
            if proc.poll() is None:
                logger.info("Sending SIGTERM to vLLM server pid=%d", proc.pid)
                proc.terminate()

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
