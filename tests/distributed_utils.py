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

import inspect
import os
import signal
import socket
import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Union

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_xla.distributed.xla_multiprocessing as xmp
from _pytest.fixtures import FixtureLookupError
from _pytest.outcomes import Skipped

from optimum.neuron.utils.cache_utils import get_num_neuron_cores
from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
    is_torch_neuronx_available,
    is_torch_xla_available,
)


if is_torch_neuronx_available():
    import torch_neuronx

if is_torch_xla_available():
    import torch_xla.distributed.xla_backend as xbn

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
    requires_neuron_environment: bool = True
    _pool_cache = {}

    @abstractmethod
    def run(self): ...

    def __call__(self, request=None):
        self._fixture_kwargs = self._get_fixture_kwargs(request, self.run)
        world_size = self.world_size
        if self.requires_neuron_environment and not is_neuron_environment_available():
            pytest.skip("Only supported in a Neuron environment.")

        # The function to be run is passed with its parameters
        run_fn = partial(self.run, **self._fixture_kwargs)

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            launch_procs(run_fn, procs, self.tp_size, self.pp_size)

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


def launch_procs(run_fn, num_procs, tp_size, pp_size, timeout = TEST_TIMEOUT):
    if not is_torch_neuronx_available() or not is_torch_xla_available() or not is_neuronx_distributed_available():
        raise RuntimeError(
            "The `torch_neuronx`, `torch_xla` and `neuronx_distributed` packages are required to run a distributed "
            "test."
        )

    # Verify we have enough accelerator devices to run this test
    num_cores = get_num_neuron_cores()
    if 0 < num_cores < num_procs:
        pytest.skip(
            f"Skipping test because not enough Neuron cores are available: {num_procs} required, {num_cores} "
            "available."
        )

    # Set start method to `forkserver` (or `fork`)
    mp.set_start_method("forkserver", force=True)

    # We cannot set environment variable `TORCHELASTIC_RUN_ID` here because `torch_neuronx` will
    # configure PJRT if it is set. Instead we create the value and set it once the other environment
    # variables to simulate a `torchrun` execution (e.g. `LOCAL_RANK`, `RANK`, `WORLD_SIZE`, ...) can be set.
    run_id = str(uuid.uuid4())

    # Create process pool or use cached one
    master_port = None
    pool = mp.Pool(processes=num_procs)
    master_port = get_master_port()

    # Run the test
    args = [(run_fn, run_id, local_rank, num_procs, master_port, tp_size, pp_size) for local_rank in range(num_procs)]
    skip_msgs_async = pool.starmap_async(_dist_run, args)

    skip_msgs = ""  # Otherwise the linter complains.
    try:
        skip_msgs = skip_msgs_async.get(timeout)
    except mp.TimeoutError:
        pytest.fail("Test hanged, skipping")
    except Exception as e:
        _close_pool(pool, num_procs, use_terminate=True)
        raise e
    finally:
        # Tear down distributed environment and close process pools
        _close_pool(pool, num_procs)

    # If we skipped a test, propagate that to this process
    if any(skip_msgs):
        assert len(set(skip_msgs)) == 1, "Multiple different skip messages received"
        pytest.skip(skip_msgs[0])


def _dist_run(run_fn, run_id, local_rank, num_procs, master_port, tp_size, pp_size):
    skip_msg = ""
    if not dist.is_initialized():
        """Initializes communication and executes the user function."""
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = master_port
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_SIZE"] = str(num_procs)
        os.environ["WORLD_SIZE"] = str(num_procs)
        os.environ["LOCAL_WORLD_SIZE"] = str(num_procs)
        # Unit tests do not support multi-node so there is only one group in our case
        os.environ["GROUP_RANK"] = "0"

        os.environ["TORCHELASTIC_RUN_ID"] = run_id

        # Now that the environment has been set, we can initialize the XLA environment.
        torch_neuronx.initialization.initialize()

        dist.init_process_group(backend="xla", rank=local_rank, world_size=num_procs)
        # dist.init_process_group(backend="xla")
        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")

        # Initializing NxD.
        neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )
    try:
        run_fn()
    except BaseException as e:
        if isinstance(e, Skipped):
            skip_msg = e.msg
        else:
            raise e

    return skip_msg


def _close_pool(pool, num_procs, use_terminate=False):
    _ = pool.starmap(_dist_destroy, [() for _ in range(num_procs)])
    if use_terminate:
        pool.terminate()
    else:
        pool.close()
    pool.join()
    _kill_neuron_processes()

def _dist_destroy():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def _kill_neuron_processes():
    try:
        cmd = "neuron-ls | grep -oE '[0-9]{7}' | xargs -r kill -9"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Command to kill all processes on Neuron failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"Error executing command: {e}")


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

    def run(self, **fixture_kwargs):
        self._current_test(**fixture_kwargs)

    def __call__(self, request):
        self._current_test = self._get_current_test_func(request)
        self._fixture_kwargs = self._get_fixture_kwargs(request, self._current_test)

        if self.requires_neuron_environment and not is_neuron_environment_available():
            pytest.skip("Only supported in a Neuron environment.")

        self.early_skip(self._fixture_kwargs)
        # The function to be run is passed with its parameters
        run_fn = partial(self.run, **self._fixture_kwargs)

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
            parallel_sizes = try_to_override_via_pytest_mark(mark, "parallel_sizes")

        # Catch world_size, tp_size or pp_size override via fixture.
        def try_to_override_via_fixture(name, current_value):
            if name in self._fixture_kwargs:
                if current_value is not None:
                    raise ValueError(f"It is not possible to override {name} both via pytest.mark and fixtures.")
                return self._fixture_kwargs[name]
            return current_value

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
            launch_procs(run_fn, *sizes)
            time.sleep(0.5)

    def _get_current_test_func(self, request):
        # DistributedTest subclasses may have multiple test methods
        func_name = request.function.__name__
        return getattr(self, func_name)

import functools
from typing import Any, Callable, Dict, Tuple

import cloudpickle
import neuronx_distributed
import torch_neuronx


class PickableSkipped(BaseException):
    """
    A picklable version of Skipped exception to be used in distributed tests.
    This is necessary because the original Skipped exception cannot be pickled.
    """
    def __init__(self, msg: str):
        self.msg = msg

    def __reduce__(self):
        return (self.__class__, (self.msg,))


def get_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)

def _distributed_worker(index: int, func_bytes: Callable, func_args: Tuple[Any, ...], func_kwargs: Dict[str, Any], master_port: str, world_size: int, tp_size: int, pp_size: int):
    rank = index  # In xmp.spawn, index is the rank of the process
    try:
        func = cloudpickle.loads(func_bytes)
        # Set up environment variables to emulate torchrun
        env = {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            # It is important to use the same port for all processes
            "MASTER_PORT": master_port,
            "GROUP_RANK": "0",
            "TORCHELASTIC_RESTART_COUNT": "0",
            "TORCHELASTIC_MAX_RESTARTS": "0",
            "TORCHELASTIC_RUN_ID": f"test_{hash(func_bytes)}"
        }
        os.environ.update(env)

        # Now that the environment has been set, we can initialize the XLA environment.
        torch_neuronx.initialization.initialize()

        dist.init_process_group(backend="xla")

        # Initializing NxD.
        neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )

        func(*func_args, **func_kwargs)

    except Skipped as e:
        raise PickableSkipped(e.msg)
    except Exception as e:
        raise e
    finally:
        # Ensure that the process group is destroyed after the test
        if dist.is_initialized():
            dist.destroy_process_group()
    return {"status": "success", "message": "Test completed successfully"}

def distributed_test(world_size: Optional[int] = None, tp_size: Optional[int] = None, pp_size: Optional[int] = None, timeout: int = 600):
    """
    Decorator to run a test function in a distributed setting.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If the test is marked as skip, skip it immediately.
            if isinstance(func, Skipped):
                raise func

            nonlocal world_size, tp_size, pp_size
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if bound.arguments.get("world_size", None) is not None:
                actual_world_size = bound.arguments.get('world_size')
            elif world_size is not None:
                actual_world_size = world_size
            else:
                actual_world_size = 1

            if bound.arguments.get("tp_size", None) is not None:
                actual_tp_size = bound.arguments.get('tp_size')
            elif tp_size is not None:
                actual_tp_size = tp_size
            else:
                actual_tp_size = 1

            if bound.arguments.get("pp_size", None) is not None:
                actual_pp_size = bound.arguments.get('pp_size')
            elif pp_size is not None:
                actual_pp_size = pp_size
            else:
                actual_pp_size = 1

            # Make the function serializable with cloudpickle
            func_bytes = cloudpickle.dumps(func)

            # This environment variable controls the number of Neuron cores used by the test.
            os.environ["NEURONCORE_NUM_DEVICES"] = str(actual_world_size)

            # There are two values allowed for `nprocs`:
            #   - `nprocs = None` means that xmp.spawn will spawn as many processes as the number of Neuron cores
            #   available, e.g. NEURONCORE_NUM_DEVICES=8 will spawn 8 processes.
            #   - `nprocs = 1` means that xmp.spawn will spawn only one process.
            nprocs = 1 if actual_world_size == 1 else None

            master_port = get_free_port()

            # Setting the timeout feature
            def timeout_handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            try:
                xmp.spawn(
                    _distributed_worker,
                    args=(func_bytes, bound.args, bound.kwargs, master_port, actual_world_size, actual_tp_size, actual_pp_size),
                    nprocs=nprocs,
                    join=True,
                )
            except TimeoutError:
                pytest.fail(f"Test timed out after {timeout}s")
            except PickableSkipped as e:
                pytest.skip(e.msg)
            except Exception as e:
                pytest.fail(str(e))
            finally:
                signal.alarm(0)

        return wrapper
    return decorator


def run_distributed_test(func: Callable, world_size: int = 1, tp_size: int = 1, pp_size: int = 1, timeout: int = 600):
    return distributed_test(world_size=world_size, tp_size=tp_size, pp_size=pp_size, timeout=timeout)(func)()
