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
"""Unit tests for VLLMServerManager core partitioning and lifecycle."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from optimum.neuron.vllm.server_manager import VLLMServerManager


@pytest.fixture
def manager():
    """A manager with 2 DP replicas, each using 2 cores (tp_size=2)."""
    return VLLMServerManager(ports=[8081, 8082], tp_size=2, vllm_args=[])


# --- _resolve_physical_cores ---


def test_resolve_physical_cores_unset(monkeypatch):
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    assert VLLMServerManager._resolve_physical_cores() is None


def test_resolve_physical_cores_range(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "4-7")
    assert VLLMServerManager._resolve_physical_cores() == [4, 5, 6, 7]


def test_resolve_physical_cores_comma(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "2,5,6,7")
    assert VLLMServerManager._resolve_physical_cores() == [2, 5, 6, 7]


def test_resolve_physical_cores_single(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "3")
    assert VLLMServerManager._resolve_physical_cores() == [3]


def test_resolve_physical_cores_whitespace(monkeypatch):
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", " 1 , 2 , 3 ")
    assert VLLMServerManager._resolve_physical_cores() == [1, 2, 3]


# --- _core_range ---


def test_core_range_no_env_var(monkeypatch, manager):
    """Without NEURON_RT_VISIBLE_CORES, cores start from 0."""
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    assert manager._core_range(0) == "0-1"
    assert manager._core_range(1) == "2-3"


def test_core_range_range_format(monkeypatch, manager):
    """With range '4-7', rank 0 gets '4-5', rank 1 gets '6-7'."""
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "4-7")
    assert manager._core_range(0) == "4-5"
    assert manager._core_range(1) == "6-7"


def test_core_range_comma_format(monkeypatch, manager):
    """With comma list '2,5,6,7', rank 0 gets '2,5', rank 1 gets '6-7'."""
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "2,5,6,7")
    assert manager._core_range(0) == "2,5"
    assert manager._core_range(1) == "6-7"


def test_core_range_insufficient_cores(monkeypatch, manager):
    """Raises ValueError when not enough cores for the requested rank."""
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "4-5")
    with pytest.raises(ValueError, match="only.*available"):
        manager._core_range(1)


def test_core_range_single_core_tp1(monkeypatch):
    """TP=1 with single core constraint."""
    mgr = VLLMServerManager(ports=[8081], tp_size=1, vllm_args=[])
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "3")
    assert mgr._core_range(0) == "3"


# --- start ---


def _make_mock_proc(pid=1000, poll_return=None):
    """Create a mock Popen process."""
    proc = MagicMock()
    proc.pid = pid
    proc.poll.return_value = poll_return
    proc.wait.return_value = poll_return
    return proc


@patch("optimum.neuron.vllm.server_manager.subprocess.Popen")
def test_start_spawns_correct_number(mock_popen, monkeypatch):
    """start() launches one subprocess per port."""
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    mock_popen.return_value = _make_mock_proc()
    mgr = VLLMServerManager(ports=[9001, 9002, 9003], tp_size=1, vllm_args=["--model", "m"])
    mgr.start()
    assert mock_popen.call_count == 3
    assert len(mgr._processes) == 3


@patch("optimum.neuron.vllm.server_manager.subprocess.Popen")
def test_start_passes_correct_ports(mock_popen, monkeypatch):
    """Each subprocess receives its assigned port."""
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    mock_popen.return_value = _make_mock_proc()
    mgr = VLLMServerManager(ports=[9001, 9002], tp_size=1, vllm_args=[])
    mgr.start()
    for call_args, expected_port in zip(mock_popen.call_args_list, [9001, 9002]):
        cmd = call_args[0][0]
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == str(expected_port)


@patch("optimum.neuron.vllm.server_manager.subprocess.Popen")
def test_start_sets_core_env(mock_popen, monkeypatch):
    """Each subprocess gets NEURON_RT_VISIBLE_CORES for its rank."""
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    mock_popen.return_value = _make_mock_proc()
    mgr = VLLMServerManager(ports=[9001, 9002], tp_size=2, vllm_args=[])
    mgr.start()
    envs = [call_args[1]["env"]["NEURON_RT_VISIBLE_CORES"] for call_args in mock_popen.call_args_list]
    assert envs == ["0-1", "2-3"]


@patch("optimum.neuron.vllm.server_manager.subprocess.Popen")
def test_start_passes_vllm_args(mock_popen, monkeypatch):
    """Extra vllm_args are appended to the subprocess command."""
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    mock_popen.return_value = _make_mock_proc()
    extra = ["--model", "my-model", "--dtype", "bfloat16"]
    mgr = VLLMServerManager(ports=[9001], tp_size=1, vllm_args=extra)
    mgr.start()
    cmd = mock_popen.call_args[0][0]
    assert cmd[-len(extra) :] == extra


# --- monitor ---


def test_monitor_returns_exit_code():
    """monitor() returns the exit code of the first process that exits."""
    mgr = VLLMServerManager(ports=[9001], tp_size=1, vllm_args=[])
    proc = _make_mock_proc(pid=100)
    proc.poll.side_effect = [None, 42]
    mgr._processes = [proc]
    assert mgr.monitor() == 42


def test_monitor_detects_first_failure():
    """With multiple processes, monitor() returns as soon as one exits."""
    mgr = VLLMServerManager(ports=[9001, 9002], tp_size=1, vllm_args=[])
    proc1 = _make_mock_proc(pid=100, poll_return=None)
    proc2 = _make_mock_proc(pid=200, poll_return=1)
    mgr._processes = [proc1, proc2]
    assert mgr.monitor() == 1


# --- shutdown ---


def test_shutdown_terminates_and_clears():
    """shutdown() terminates running processes and clears the list."""
    mgr = VLLMServerManager(ports=[9001], tp_size=1, vllm_args=[])
    proc = _make_mock_proc(pid=100, poll_return=None)
    proc.wait.return_value = 0
    proc.poll.side_effect = [None, 0]
    mgr._processes = [proc]

    mgr.shutdown()

    proc.terminate.assert_called_once()
    assert len(mgr._processes) == 0
    assert mgr._stop.is_set()


def test_shutdown_force_kills_on_timeout():
    """shutdown() sends SIGKILL when a process doesn't exit after terminate."""
    mgr = VLLMServerManager(ports=[9001], tp_size=1, vllm_args=[])
    proc = _make_mock_proc(pid=100)
    proc.poll.return_value = None
    proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 10), None]
    mgr._processes = [proc]

    mgr.shutdown()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()


def test_shutdown_skips_already_exited():
    """shutdown() does not terminate processes that already exited."""
    mgr = VLLMServerManager(ports=[9001], tp_size=1, vllm_args=[])
    proc = _make_mock_proc(pid=100, poll_return=0)
    mgr._processes = [proc]

    mgr.shutdown()

    proc.terminate.assert_not_called()
