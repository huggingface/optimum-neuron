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

import functools
import inspect
import os
import signal
import socket
import sys
import time
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

import cloudpickle
import psutil
import pytest
import torch.distributed as dist
import torch_xla.distributed.xla_multiprocessing as xmp
from _pytest.outcomes import Skipped

from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
)


if is_neuronx_distributed_available():
    import neuronx_distributed

TEST_TIMEOUT = 600


class PicklableException(BaseException):
    """
    A wrapper that can serialize any exception with its traceback.
    Works across process boundaries where original exceptions might fail to pickle.
    """

    def __init__(self, exc_type: str, exc_value: str, tb_str: str):
        self.exc_type_name = exc_type
        self.exc_value = exc_value
        self.tb_str = tb_str
        super().__init__(f"{self.exc_type_name}: {exc_value}")

    def __reduce__(self):
        # This method enables pickling
        return (self.__class__, (self.exc_type_name, self.exc_value, self.tb_str))

    @classmethod
    def from_exception(cls, exc: BaseException):
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(type(exc).__name__, str(exc), tb_str)


def get_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)


def _distributed_worker(
    index: int,
    func_bytes: Callable,
    func_args: Tuple[Any, ...],
    func_kwargs: Dict[str, Any],
    master_port: str,
    world_size: int,
    tp_size: int,
    pp_size: int,
):
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
            "TORCHELASTIC_RUN_ID": f"test_{hash(func_bytes)}",
        }
        os.environ.update(env)

        dist.init_process_group(backend="xla")

        neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
        )

        func(*func_args, **func_kwargs)

    except BaseException as e:
        # Catch all exceptions and wrap them in a PicklableException
        print(f"Exception in process {rank}: {e}")
        # raise PicklableException.from_exception(e)
        sys.exit(1)
    finally:
        # Ensure that the process group is destroyed after the test
        if dist.is_initialized():
            dist.destroy_process_group()


def _terminate_processes():
    print("Terminating child processes...")
    current_process = psutil.Process()
    children = list(current_process.children(recursive=True))
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    # Wait briefly for graceful termination
    time.sleep(2)

    # Force kill if still alive
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass


def distributed_test(
    world_size: Optional[int] = None, tp_size: Optional[int] = None, pp_size: Optional[int] = None, timeout: int = 600
):
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
                actual_world_size = bound.arguments.get("world_size")
            elif world_size is not None:
                actual_world_size = world_size
            else:
                actual_world_size = 1

            if bound.arguments.get("tp_size", None) is not None:
                actual_tp_size = bound.arguments.get("tp_size")
            elif tp_size is not None:
                actual_tp_size = tp_size
            else:
                actual_tp_size = 1

            if bound.arguments.get("pp_size", None) is not None:
                actual_pp_size = bound.arguments.get("pp_size")
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
            # Here we set `nprocs` to None and set `NEURONCORE_NUM_DEVICES` to the number of processes we want to spawn.
            nprocs = None

            master_port = get_free_port()

            # Setting the timeout feature
            def timeout_handler(signum, frame):
                raise TimeoutError()

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            try:
                xmp.spawn(
                    _distributed_worker,
                    args=(
                        func_bytes,
                        bound.args,
                        bound.kwargs,
                        master_port,
                        actual_world_size,
                        actual_tp_size,
                        actual_pp_size,
                    ),
                    nprocs=nprocs,
                    join=True,
                )
            except TimeoutError:
                pytest.fail(f"Test timed out after {timeout}s", pytrace=False)
            except PicklableException as e:
                if e.exc_type_name == "Skipped":
                    pytest.skip(e.exc_value)
                elif e.exc_type_name == "XFailed":
                    pytest.xfail(e.exc_value)
                else:
                    print(f"\n{'=' * 60}")
                    print("Exception from distributed process:")
                    print(f"{'=' * 60}")
                    print(e.tb_str)
                    print(f"{'=' * 60}\n")
                    pytest.fail(f"Test failed with {e.exc_type_name}: {e.exc_value}", pytrace=False)
            finally:
                _terminate_processes()
                signal.alarm(0)

        return wrapper

    return decorator


def run_distributed_test(func: Callable, world_size: int = 1, tp_size: int = 1, pp_size: int = 1, timeout: int = 600):
    return distributed_test(world_size=world_size, tp_size=tp_size, pp_size=pp_size, timeout=timeout)(func)()
