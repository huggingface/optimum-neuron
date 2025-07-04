# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import math
import sys
import time

import pytest
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from .distributed_utils import EarlyExit, distributed_test


@pytest.mark.parametrize(
    "exit_code",
    [None, 0, 1, 2],
)
@distributed_test(world_size=2)
@pytest.mark.xfail(reason="This test is expected to fail on some processes")
def test_sys_exit(exit_code):
    print("Running a test that raises SystemExit (normal exit path)")
    if xr.global_ordinal() == 0:
        print("Raising SystemExit on process 0")
        sys.exit(exit_code)


@distributed_test(world_size=32)
@pytest.mark.xfail(reason="This test is expected to fail on some processes")
def test_raise_exception_while_other_processes_are_waiting_for_rendezvous():
    local_world_size = xr.world_size()
    num_local_ranks_per_step = 8
    local_rank = xr.local_ordinal()
    for worker in range(math.ceil(local_world_size / num_local_ranks_per_step)):
        if local_rank // num_local_ranks_per_step == worker:
            print(f"Worker #{local_rank} is raising an exception")
            raise RuntimeError(f"Simulated error on worker {worker} with local rank {local_rank}")
        print(f"Worker #{local_rank} is waiting for rendezvous")
        xm.rendezvous(f"load_state_dict_{local_rank}")


@distributed_test(world_size=2)
def test_skip():
    print("Running a test that skips the rest of the test")
    if xr.global_ordinal() == 0:
        print("Skipping the rest of the test on process 0")
        pytest.skip("Skipping this test for demonstration purposes")
    else:
        print("Process 1 continues with the test")


@distributed_test(world_size=2)
@pytest.mark.xfail(reason="This test is expected to fail on process 0")
def test_xfail():
    print("Running a test that is expected to fail")
    if xr.global_ordinal() == 0:
        print("Raising an exception on process 0")
        raise RuntimeError("This test is expected to fail on process 0")
    else:
        print("Process 1 continues with the test")


@distributed_test(world_size=2, timeout=2)
@pytest.mark.xfail(reason="This test is expected to timeout")
def test_timeout():
    time.sleep(5)  # This will cause a timeout if the test is not skipped or failed


@distributed_test(world_size=2)
def test_succeed():
    print("Running a test that succeeds")
    assert True, "This test should succeed"
    print("Test succeeded")


@distributed_test(world_size=2)
@pytest.mark.parametrize("exit_code", [0, pytest.param(1, marks=pytest.mark.xfail(reason="Expected failure"))])
def test_succeed_with_early_exit(exit_code):
    print("Running a test that succeeds")
    if xr.global_ordinal() == 0:
        print(f"Raising EarlyExit with exit code {exit_code} on process 0")
        raise EarlyExit(exit_code)
    time.sleep(600)  # This should not be executed if EarlyExit is raised
    print("Work done")
