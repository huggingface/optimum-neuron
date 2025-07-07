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

import concurrent
import functools
import inspect
import os
import socket
import time
import traceback
from typing import Any, Callable, TypeVar

import cloudpickle
import neuronx_distributed
import pytest
import torch
import torch.distributed as dist
import torch_xla
from _pytest.outcomes import Exit, Skipped, XFailed
from torch_xla import runtime
from torch_xla._internal import neuron
from torch_xla._internal.pjrt import _merge_replica_results, _run_thread_per_device, initialize_multiprocess


TEST_TIMEOUT = 600

R = TypeVar("R")


class EarlyExit(Exception):
    """
    Exception to indicate that the distributed test should exit early.
    `returncode=0` or `returncode=None` indicates success, while any other values indicate failure.
    """

    def __init__(self, msg: str, returncode: int = 0):
        super().__init__(msg)
        self.msg = msg
        self.returncode = returncode


class PicklableException(Exception):
    def __init__(self, exc_type: str, exc_value: str, tb_str: str, exception_attributes: dict[str, Any | None] = None):
        self.exc_type_name = exc_type
        self.exc_value = exc_value
        self.tb_str = tb_str
        self.exception_attributes = {}
        if exception_attributes is not None:
            for key, value in exception_attributes.items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    self.exception_attributes[key] = value
        super().__init__(f"{self.exc_type_name}: {exc_value}")

    def __reduce__(self):
        return (
            self.__class__,
            (self.exc_type_name, self.exc_value, self.tb_str, self.exception_attributes),
        )

    @classmethod
    def from_exception(cls, exc: Exception):
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(type(exc).__name__, str(exc), tb_str, exc.__dict__)


def _termintate_executor_processes(
    executor: concurrent.futures.ProcessPoolExecutor, futures: list[concurrent.futures.Future]
):
    # Cancel all futures in the executor
    # This will cancel only the pending tasks, not the ones that are already running.
    for future in futures:
        future.cancel()

    # Terminate all processes in the executor
    for process in executor._processes.values():
        if not process.is_alive():
            continue
        process.terminate()
    time.sleep(2)  # Allow time for processes to terminate gracefully
    for process in executor._processes.values():
        if not process.is_alive():
            continue
        process.kill()

    # Shutdown the executor, it will free all the resources used by the executor
    executor.shutdown(wait=False)


def run_multiprocess(
    fn: Callable[..., R], *args, start_method: str = "spawn", timeout: int = TEST_TIMEOUT, **kwargs
) -> dict[int, R]:
    """
    Runs `fn` on all devices available to PjRt.
    Spawns one process per physical device (e.g. Neuron device).

    Args:
        fn: Function to run on all devices
        args: args to pass to `fn`
        start_method: The Python `multiprocessing` process creation method.
          Default: `spawn`
        timeout: Timeout for the test in seconds.
         kwargs: kwargs to pass to `fn`

    Returns:
      Dict of the form {device_ordinal: return_value}, where return_value is the result of calling `fn`.
    """
    if torch_xla._XLAC._xla_runtime_is_initialized():
        raise RuntimeError("Runtime is already initialized. Do not use the XLA device before calling xmp.spawn.")

    if runtime.device_type() == "NEURON":
        num_processes = neuron.num_local_processes()
    else:
        num_processes = 1

    replica_results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_processes,
        mp_context=torch.multiprocessing.get_context(start_method),
    ) as executor:
        mp_fn = functools.partial(
            _run_thread_per_device,
            local_world_size=num_processes,
            fn=functools.partial(fn, *args, **kwargs),
            initializer_fn=initialize_multiprocess,
        )
        futures = [executor.submit(mp_fn, i) for i in range(num_processes)]
        future_to_ordinal = {future: i for i, future in enumerate(futures)}

        try:
            for future in concurrent.futures.as_completed(futures, timeout=timeout):
                ordinal = future_to_ordinal[future]
                try:
                    result = future.result()
                    replica_results.append((ordinal, result))
                except PicklableException as e:
                    if e.exc_type_name == "Skipped":
                        pytest.skip(e.exception_attributes["msg"])
                    elif e.exc_type_name == "XFailed":
                        pytest.xfail(e.exception_attributes["msg"], pytrace=False)
                    elif e.exc_type_name == "Exit":
                        pytest.exit(
                            e.exception_attributes["msg"],
                            returncode=e.exception_attributes["returncode"],
                            pytrace=False,
                        )
                    elif e.exc_type_name == "EarlyExit" and e.exception_attributes["returncode"] in (0, None):
                        replica_results.append((ordinal, None))
                        # We need to break since we do not raise anything here.
                        # If we don't, we will wait for all processes to finish.
                        break
                    else:
                        print(f"\n{'=' * 80}")
                        print("Exception from distributed process:")
                        print(f"{'=' * 80}")
                        print(e.tb_str)
                        print(f"{'=' * 80}\n")
                        pytest.fail(f"Test failed with {e.exc_type_name}: {e.exc_value}", pytrace=False)
        except concurrent.futures.TimeoutError:
            pytest.fail(f"Test timed out after {timeout}s", pytrace=False)
        finally:
            _termintate_executor_processes(executor, futures)

    return _merge_replica_results(replica_results)


def get_free_port():
    """Find a free port on localhost"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return str(port)


def _distributed_worker(
    func_bytes: Callable,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
    master_port: str,
    world_size: int,
    tp_size: int,
    pp_size: int,
):
    rank = runtime.local_ordinal()
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

    except (Skipped, XFailed, Exit, Exception) as e:
        raise PicklableException.from_exception(e)
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as dist_e:
            print(f"Failed to destroy process group: {dist_e}")


def distributed_test(
    world_size: int | None = None, tp_size: int | None = None, pp_size: int | None = None, timeout: int = 600
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
            master_port = get_free_port()

            run_multiprocess(
                _distributed_worker,
                func_bytes,
                bound.args,
                bound.kwargs,
                master_port,
                actual_world_size,
                actual_tp_size,
                actual_pp_size,
                timeout=timeout,
            )

        return wrapper

    return decorator


def run_distributed_test(func: Callable, world_size: int = 1, tp_size: int = 1, pp_size: int = 1, timeout: int = 600):
    return distributed_test(world_size=world_size, tp_size=tp_size, pp_size=pp_size, timeout=timeout)(func)()
