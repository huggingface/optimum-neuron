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
"""Utilities for tests distributed."""

import functools
import inspect
import os
import socket
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Type, Union

import neuronx_distributed
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from _pytest.fixtures import FixtureFunctionMarker, FixtureLookupError
from _pytest.outcomes import Skipped
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
)

from optimum.neuron.utils.cache_utils import get_num_neuron_cores
from optimum.neuron.utils.patching import DynamicPatch, Patcher
from optimum.neuron.utils.require_utils import requires_neuronx_distributed, requires_torch_xla


if TYPE_CHECKING:
    from transformers import PreTrainedModel

TEST_TIMEOUT = 600


def is_neuron_environment_available() -> bool:
    return get_num_neuron_cores() > 0


# The following code related to distributed test is copied from the DeepSpeed repo:
# https://github.com/microsoft/DeepSpeed/blob/master/tests/unit/common.py


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
    methods needed for DistributedTest and DistributedFixture.
    """

    world_size: int = 2
    tp_size: int = 1
    pp_size: int = 1
    backend: str = "xla"
    init_distributed: bool = True
    set_dist_env: bool = True
    requires_neuron_environment = True
    reuse_dist_env = False
    _pool_cache = {}
    exec_timeout = TEST_TIMEOUT

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
            self._launch_procs(procs)

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

    def _launch_procs(self, num_procs):
        # Verify we have enough accelerator devices to run this test
        num_cores = get_num_neuron_cores()
        if 0 < num_cores < num_procs:
            pytest.skip(
                f"Skipping test because not enough Neuron cores are available: {num_procs} required, {num_cores} "
                "available."
            )

        # Set start method to `forkserver` (or `fork`)
        mp.set_start_method("forkserver", force=True)

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
        args = [(local_rank, num_procs, master_port) for local_rank in range(num_procs)]
        skip_msgs_async = pool.starmap_async(self._dist_run, args)

        try:
            skip_msgs = skip_msgs_async.get(self.exec_timeout)
        except mp.TimeoutError:
            # Shortcut to exit pytest in the case of a hanged test. This
            # usually means an environment error and the rest of tests will
            # hang (causing super long unit test runtimes)
            pytest.exit("Test hanged, exiting", returncode=0)

        # Tear down distributed environment and close process pools
        self._close_pool(pool, num_procs)

        # If we skipped a test, propagate that to this process
        if any(skip_msgs):
            assert len(set(skip_msgs)) == 1, "Multiple different skip messages received"
            pytest.skip(skip_msgs[0])

    def _dist_run(self, local_rank, num_procs, master_port):
        skip_msg = ""
        if not dist.is_initialized():
            """Initializes communication and executes the user function."""
            if self.set_dist_env:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = str(master_port)
                os.environ["LOCAL_RANK"] = str(local_rank)
                # NOTE: unit tests don't support multi-node so local_rank == global rank
                os.environ["RANK"] = str(local_rank)
                os.environ["LOCAL_SIZE"] = str(num_procs)
                os.environ["WORLD_SIZE"] = str(num_procs)

        if self.init_distributed:
            # Initializing the process group.
            dist.init_process_group(backend=self.backend, rank=local_rank, world_size=self.world_size)
            dist.barrier()

            # Intializing NxD.
            neuronx_distributed.parallel_layers.parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                pipeline_model_parallel_size=self.tp_size,
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
            _ = pool.starmap(self._dist_destroy, [() for _ in range(num_procs)])
            pool.close()
            pool.join()


class DistributedFixture(DistributedExec):
    """
    Implementation that extends @pytest.fixture to allow for distributed execution.
    This is primarily meant to be used when a test requires executing two pieces of
    code with different world sizes.

    There are 2 parameters that can be modified:
        - world_size: int = 2 -- the number of processes to launch
        - backend: Literal['nccl','mpi','gloo'] = 'nccl' -- which backend to use

    Features:
        - able to call pytest.skip() inside fixture
        - can be reused by multiple tests
        - can accept other fixtures as input

    Limitations:
        - cannot use @pytest.mark.parametrize
        - world_size cannot be modified after definition and only one world_size value is accepted
        - any fixtures used must also be used in the test that uses this fixture (see example below)
        - return values cannot be returned. Passing values to a DistributedTest
          object can be achieved using class_tmpdir and writing to file (see example below)

    Usage:
        - must implement a run(self, ...) method
        - fixture can be used by making the class name input to a test function

    Example:
        @pytest.fixture(params=[10,20])
        def regular_pytest_fixture(request):
            return request.param

        class distributed_fixture_example(DistributedFixture):
            world_size = 4

            def run(self, regular_pytest_fixture, class_tmpdir):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                local_rank = os.environ["LOCAL_RANK"]
                print(f"Rank {local_rank} with value {regular_pytest_fixture}")
                with open(os.path.join(class_tmpdir, f"{local_rank}.txt"), "w") as f:
                    f.write(f"{local_rank},{regular_pytest_fixture}")

        class TestExample(DistributedTest):
            world_size = 1

            def test(self, distributed_fixture_example, regular_pytest_fixture, class_tmpdir):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                for rank in range(4):
                    with open(os.path.join(class_tmpdir, f"{rank}.txt"), "r") as f:
                        assert f.read() == f"{rank},{regular_pytest_fixture}"
    """

    is_dist_fixture = True

    # These values are just placeholders so that pytest recognizes this as a fixture
    _pytestfixturefunction = FixtureFunctionMarker(scope="function", params=None)
    __name__ = ""

    def __init__(self):
        assert isinstance(self.world_size, int), "Only one world size is allowed for distributed fixtures"
        self.__name__ = type(self).__name__
        _pytestfixturefunction = FixtureFunctionMarker(scope="function", params=None, name=self.__name__)


class DistributedTest(DistributedExec):
    """
    Implementation for running pytest with distributed execution.

    There are 2 parameters that can be modified:
        - world_size: Union[int,List[int]] = 2 -- the number of processes to launch
        - backend: Literal['nccl','mpi','gloo'] = 'nccl' -- which backend to use

    Features:
        - able to call pytest.skip() inside tests
        - works with pytest fixtures, parametrize, mark, etc.
        - can contain multiple tests (each of which can be parametrized separately)
        - class methods can be fixtures (usable by tests in this class only)
        - world_size can be changed for individual tests using @pytest.mark.world_size(world_size)
        - class_tmpdir is a fixture that can be used to get a tmpdir shared among
          all tests (including DistributedFixture)

    Usage:
        - class name must start with "Test"
        - must implement one or more test*(self, ...) methods

    Example:
        @pytest.fixture(params=[10,20])
        def val1(request):
            return request.param

        @pytest.mark.fast
        @pytest.mark.parametrize("val2", [30,40])
        class TestExample(DistributedTest):
            world_size = 2

            @pytest.fixture(params=[50,60])
            def val3(self, request):
                return request.param

            def test_1(self, val1, val2, str1="hello world"):
                assert int(os.environ["WORLD_SIZE"]) == self.world_size
                assert all(val1, val2, str1)

            @pytest.mark.world_size(1)
            @pytest.mark.parametrize("val4", [70,80])
            def test_2(self, val1, val2, val3, val4):
                assert int(os.environ["WORLD_SIZE"]) == 1
                assert all(val1, val2, val3, val4)
    """

    is_dist_test = True

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

        # Catch world_size override pytest mark
        for mark in getattr(request.function, "pytestmark", []):
            if mark.name == "world_size":
                world_size = mark.args[0]
                break
        else:
            world_size = self.world_size

        if isinstance(world_size, int):
            world_size = [world_size]
        for procs in world_size:
            self._launch_procs(procs)
            time.sleep(0.5)

    def _get_current_test_func(self, request):
        # DistributedTest subclasses may have multiple test methods
        func_name = request.function.__name__
        return getattr(self, func_name)


def generate_dummy_labels(
    model: "PreTrainedModel",
    shape: List[int],
    vocab_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """Generates dummy labels."""

    model_class_name = model.__class__.__name__
    labels = {}

    batch_size = shape[0]

    if model_class_name in [
        *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
        *get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES),
        *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
        *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
        *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
    ]:
        labels["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif model_class_name in [
        *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
        *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
        "XLNetForQuestionAnswering",
    ]:
        labels["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif model_class_name in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
        if not hasattr(model.config, "problem_type") or model.config.problem_type is None:
            raise ValueError(
                "Could not retrieve the problem type for the sequence classification task, please set "
                'model.config.problem_type to one of the following values: "regression", '
                '"single_label_classification", or "multi_label_classification".'
            )

        if model.config.problem_type == "regression":
            labels_shape = (batch_size, model.config.num_labels)
            labels_dtype = torch.float32
        elif model.config.problem_type == "single_label_classification":
            labels_shape = (batch_size,)
            labels_dtype = torch.long
        elif model.config.problem_type == "multi_label_classification":
            labels_shape = (batch_size, model.config.num_labels)
            labels_dtype = torch.float32
        else:
            raise ValueError(
                'Expected model.config.problem_type to be either: "regression", "single_label_classification"'
                f', or "multi_label_classification", but "{model.config.problem_type}" was provided.'
            )
        labels["labels"] = torch.zeros(*labels_shape, dtype=labels_dtype, device=device)

    elif model_class_name in [
        *get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES),
        *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
        *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES),
        "GPT2DoubleHeadsModel",
        "PeftModelForCausalLM",
        "PeftModelForSeq2SeqLM",
    ]:
        if vocab_size is None:
            raise ValueError(
                "The vocabulary size needs to be specified to generte dummy labels for language-modeling tasks."
            )
        if seed is not None:
            orig_seed = torch.seed()
            torch.manual_seed(seed)
        random_labels = torch.randint(0, vocab_size, shape, dtype=torch.long)
        if device is not None:
            random_labels = random_labels.to(device)
        labels["labels"] = random_labels
        if seed is not None:
            torch.manual_seed(orig_seed)
    elif model_class_name in [*get_values(MODEL_FOR_CTC_MAPPING_NAMES)]:
        labels["labels"] = torch.zeros(shape, dtype=torch.float32, device=device)
    else:
        raise NotImplementedError(f"Generating the dummy input named for {model_class_name} is not supported yet.")
    return labels


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )

    """Gathers tensors and concatenate along the last dimension."""

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 device.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Compilation fails when not synchronizing here.
    xm.mark_step()

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    xm.mark_step()

    return output


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_size,
    )

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output = xm.all_gather(input_, groups=get_tensor_model_parallel_group(as_list=True), pin_layout=False)

    return output


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_dim(input_: torch.Tensor, dim: int) -> torch.Tensor:
    input_ = input_.clone().contiguous()
    if dim == 0:
        return gather_along_first_dim(input_)
    elif dim in [-1, input_.dim() - 1]:
        return gather_along_last_dim(input_)
    else:
        t = input_.transpose(0, dim).contiguous()
        gathered_t = gather_along_first_dim(t)
        return gathered_t.transpose(0, dim).contiguous()


def static_initializer_seed(initialization_function: Callable, seed: int):
    @functools.wraps(initialization_function)
    def wrapper(*args, **kwargs):
        from transformers import set_seed

        set_seed(seed)
        return initialization_function(*args, **kwargs)

    return wrapper


@contextmanager
def create_static_seed_patcher(model_class: Type["PreTrainedModel"], seed: int):
    """
    Context manager that resets the seed to a given value for every initialization function.
    This is useful because lazy initialization works but does not respect the random state of the non-lazy case.
    This allows us to test that lazy initialization works if we ignore the random seed.
    """
    specialized_static_initializer_seed = functools.partial(static_initializer_seed, seed=seed)

    class_module_name = inspect.getmodule(model_class).__name__
    fully_qualified_method_name = f"{class_module_name}.{model_class.__name__}._init_weights"
    dynamic_patch = DynamicPatch(specialized_static_initializer_seed)
    patcher = Patcher(
        [
            (fully_qualified_method_name, dynamic_patch),
            ("torch.nn.Embedding.reset_parameters", dynamic_patch),
            ("torch.nn.Linear.reset_parameters", dynamic_patch),
            ("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear.init_weight_cpu", dynamic_patch),
            ("neuronx_distributed.parallel_layers.layers.RowParallelLinear.init_weight_cpu", dynamic_patch),
        ]
    )
    with patcher:
        try:
            yield
        finally:
            pass
