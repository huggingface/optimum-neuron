# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Defines classes to enable running tests in a distributed setting."""

# The following code is copied and adapted from the DeepSpeed repo:
# https://github.com/microsoft/DeepSpeed/blob/master/tests/unit/common.py

import inspect
import multiprocessing
import os
import socket
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Union

import psutil
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_xla.distributed.xla_backend as xbn
from _pytest.fixtures import FixtureLookupError
from _pytest.outcomes import Skipped

from optimum.neuron.utils.cache_utils import get_num_neuron_cores
from optimum.neuron.utils.import_utils import is_neuronx_distributed_available

if is_neuronx_distributed_available():
    import neuronx_distributed

TEST_TIMEOUT = 600


def is_neuron_environment_available() -> bool:
    return get_num_neuron_cores() > 0


def get_xdist_worker_id():
    xdist_worker = os.environ.get("PYTEST_XDIST_WORKER", None)
    if xdist_worker is not None:
        xdist_worker_id = xdist_worker.replace("gw", "")
        return int(xdist_worker_id)
    return None


def get_master_port(base_port=29500, port_range_size=1000):
    xdist_worker_id = get_xdist_worker_id()
    if xdist_worker_id is not None:
        # Make xdist workers use different port ranges to avoid race conditions
        base_port += port_range_size * xdist_worker_id

    # Select first open port in range
    port = base_port
    max_port = base_port + port_range_size
    sock = socket.socket()
    while port < max_port:
        try:
            sock.bind(("", port))
            sock.close()
            return str(port)
        except OSError:
            port += 1
    raise IOError("no free ports")


class DistributedExec(ABC):
    """
    Base class for distributed execution of functions/methods. Contains common
    methods needed for DistributedTest and DistributedFixture (not included in this file).
    """

    world_size: Union[int, List[int]] = 2
    tp_size: int = 1
    pp_size: int = 1
    backend: str = "xla"
    init_distributed: bool = True
    set_dist_env: bool = True
    requires_neuron_environment: bool = True
    reuse_dist_env: bool = False
    _pool_cache = {}
    exec_timeout: int = TEST_TIMEOUT

    @abstractmethod
    def run(self):
        ...

    def __call__(self, request=None):
        self._fixture_kwargs = self._get_fixture_kwargs(request, self.run)
        world_size = self.world_size
        if self.requires_neuron_environment and not is_neuron_environment_available():
            pytest.skip("Only supported in a Neuron environment.")

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            self._launch_procs(procs, self.tp_size, self.pp_size)

    def _get_fixture_kwargs(self, request, func):
        if not request:
            return {}
        # Grab fixture / parametrize kwargs from pytest request object
        fixture_kwargs = {}
        params = inspect.getfullargspec(func).args
        params.remove("self")
        for p in params:
            try:
                fixture_kwargs[p] = request.getfixturevalue(p)
            except FixtureLookupError:
                pass  # test methods can have kwargs that are not fixtures
        return fixture_kwargs

    def _launch_procs(self, num_procs, tp_size, pp_size):
        if not is_neuronx_distributed_available():
            raise RuntimeError("The `neuronx_distributed` package is required to run a distributed test.")

        # Verify we have enough accelerator devices to run this test
        num_cores = get_num_neuron_cores()
        if 0 < num_cores < num_procs:
            pytest.skip(
                f"Skipping test because not enough Neuron cores are available: {num_procs} required, {num_cores} "
                "available."
            )

        # Set start method to `forkserver` (or `fork`)
        mp.set_start_method("forkserver", force=True)
        os.environ["TORCHELASTIC_RUN_ID"] = str(uuid.uuid4())

        # Create process pool or use cached one
        master_port = None
        if self.reuse_dist_env:
            if num_procs not in self._pool_cache:
                self._pool_cache[num_procs] = mp.Pool(processes=num_procs)
                master_port = get_master_port()
            pool = self._pool_cache[num_procs]
        else:
            pool = mp.Pool(processes=num_procs)
            master_port = get_master_port()

        # Run the test
        args = [(local_rank, num_procs, master_port, tp_size, pp_size) for local_rank in range(num_procs)]
        skip_msgs_async = pool.starmap_async(self._dist_run, args)

        skip_msgs = ""  # Otherwise the linter complains.
        try:
            skip_msgs = skip_msgs_async.get(self.exec_timeout)
        except mp.TimeoutError:
            # Shortcut to exit pytest in the case of a hanged test. This
            # usually means an environment error and the rest of tests will
            # hang (causing super long unit test runtimes)
            pytest.exit("Test hanged, exiting", returncode=0)
        except Exception as e:
            self._close_pool(pool, num_procs)
            self._terminate_xrt_server()
            raise e
        finally:
            # Tear down distributed environment and close process pools
            self._close_pool(pool, num_procs)
            self._terminate_xrt_server()

        # If we skipped a test, propagate that to this process
        if any(skip_msgs):
            assert len(set(skip_msgs)) == 1, "Multiple different skip messages received"
            pytest.skip(skip_msgs[0])

    def _dist_run(self, local_rank, num_procs, master_port, tp_size, pp_size):
        skip_msg = ""
        if not dist.is_initialized():
            """Initializes communication and executes the user function."""
            if self.set_dist_env:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = str(master_port)
                # Unit tests do not support multi-node so local_rank == global rank
                os.environ["LOCAL_RANK"] = str(local_rank)
                os.environ["RANK"] = str(local_rank)
                os.environ["LOCAL_SIZE"] = str(num_procs)
                os.environ["WORLD_SIZE"] = str(num_procs)
                os.environ["LOCAL_WORLD_SIZE"] = str(num_procs)
                # Unit tests do not support multi-node so there is only one group in our case
                os.environ["GROUP_RANK"] = "0"

            if self.init_distributed:
                dist.init_process_group(backend=self.backend, rank=local_rank, world_size=num_procs)
                if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
                    raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")

                # Intializing NxD.
                neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel(
                    tensor_model_parallel_size=tp_size,
                    pipeline_model_parallel_size=pp_size,
                )
        try:
            self.run(**self._fixture_kwargs)
        except BaseException as e:
            if isinstance(e, Skipped):
                skip_msg = e.msg
            else:
                raise e

        return skip_msg

    def _dist_destroy(self):
        if (dist is not None) and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    def _close_pool(self, pool, num_procs, force=False):
        if force or not self.reuse_dist_env:
            try:
                _ = pool.starmap(self._dist_destroy, [() for _ in range(num_procs)])
                pool.close()
                pool.join()
            except ValueError:
                pass

    def _terminate_xrt_server(self):
        xrt_server_str = "torch_neuronx.distributed._xrt_run_server"
        startmethod = mp.get_start_method(allow_none=True)
        # Rules:
        # - `startmethod is None`: the XRT server tracks pytest's PID.
        # - `startmethod="spawn"`: the parent process of the pool's processes is pytest, so the XRT server tracks
        # pytest's PID.
        # - `startmethod="fork"`: same as `startmethod="spawn"`.
        # - `startmethod="forkserver"`: the parent process of the pool's processes is the forkserver, so the XRT server tracks
        # the forkserver's PID.
        if startmethod == "forkserver":
            target_pid = multiprocessing.forkserver._forkserver._forkserver_pid
        else:
            target_pid = os.getpid()

        for p in psutil.process_iter():
            try:
                if "python3" in p.name() and len(p.cmdline()) == 7:
                    cmdline = p.cmdline()
                    if cmdline[2] == xrt_server_str and cmdline[-1] == str(target_pid):
                        p.terminate()
            except psutil.ZombieProcess:
                continue


class DistributedTest(DistributedExec):
    """
    Implementation for running pytest with distributed execution.
    """

    is_dist_test = True

    def early_skip(self, fixtures_kwargs):
        """
        Override to enable early test skipping (before processes creation).
        """
        pass

    # Temporary directory that is shared among test methods in a class
    @pytest.fixture(autouse=True, scope="class")
    def class_tmpdir(self, tmpdir_factory):
        fn = tmpdir_factory.mktemp(self.__class__.__name__)
        return fn

    def run(self, **fixture_kwargs):
        self._current_test(**fixture_kwargs)

    def __call__(self, request):
        self._current_test = self._get_current_test_func(request)
        self._fixture_kwargs = self._get_fixture_kwargs(request, self._current_test)

        if self.requires_neuron_environment and not is_neuron_environment_available():
            pytest.skip("Only supported in a Neuron environment.")

        self.early_skip(self._fixture_kwargs)

        world_size = tp_size = pp_size = parallel_sizes = None

        # Catch world_size, tp_size or pp_size override pytest mark.
        def try_to_override_via_pytest_mark(mark, name):
            if mark.name == name:
                return mark.args[0]
            return None

        for mark in getattr(request.function, "pytestmark", []):
            world_size = try_to_override_via_pytest_mark(mark, "world_size")
            tp_size = try_to_override_via_pytest_mark(mark, "tp_size")
            pp_size = try_to_override_via_pytest_mark(mark, "pp_size")
            parallel_sizes = try_to_override_via_pytest_mark(mark, "parallel_size")

        # Catch world_size, tp_size or pp_size override via fixture.
        def try_to_override_via_fixture(name, current_value):
            if name in self._fixture_kwargs:
                if current_value is not None:
                    raise ValueError(f"It is not possible to override {name} both via pytest.mark and fixtures.")
                return self._fixture_kwargs[name]
            return None

        world_size = try_to_override_via_fixture("world_size", world_size)
        tp_size = try_to_override_via_fixture("tp_size", tp_size)
        pp_size = try_to_override_via_fixture("pp_size", pp_size)
        parallel_sizes = try_to_override_via_fixture("parallel_sizes", parallel_sizes)

        if parallel_sizes is not None:
            if not all(size is None for size in [world_size, tp_size, pp_size]):
                raise ValueError("Either specify parallel_sizes or specific size (world_size, tp_size, pp_size)")
            world_size, tp_size, pp_size = parallel_sizes

        if world_size is None:
            world_size = self.world_size
        if tp_size is None:
            tp_size = self.tp_size
        if pp_size is None:
            pp_size = self.pp_size

        sizes = [world_size, tp_size, pp_size]
        if all(isinstance(size, int) for size in sizes):
            world_size = [world_size]
            tp_size = [tp_size]
            pp_size = [pp_size]
        else:
            lengths = [len(size) for size in sizes if not isinstance(size, int)]
            if len(set(lengths)) != 1:
                raise ValueError(
                    "When providing multiple values for either world_size, tp_size or pp_size, you must provide the "
                    f"same number of values. Here: {', '.join(lengths)}."
                )
            if not all(isinstance(size, (tuple, list)) for size in sizes):
                length = lengths[0]
                world_size = [world_size] * length if isinstance(world_size, int) else world_size
                tp_size = [tp_size] * length if isinstance(tp_size, int) else tp_size
                pp_size = [pp_size] * length if isinstance(pp_size, int) else pp_size

        for sizes in zip(world_size, tp_size, pp_size):
            self._launch_procs(*sizes)
            time.sleep(0.5)

    def _get_current_test_func(self, request):
        # DistributedTest subclasses may have multiple test methods
        func_name = request.function.__name__
        return getattr(self, func_name)
