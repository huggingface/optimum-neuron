import logging
import os
import uuid
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch_neuronx
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.trace import ModelBuilder
from neuronx_distributed.trace.model_builder import BaseModelInstance
from transformers import set_seed


logger = logging.getLogger("Neuron")


class FunctionModule(torch.nn.Module):
    """
    A module that wraps a function to run it on Neuron.
    """

    def __init__(self, func: Callable):
        super().__init__()
        self.func = func

    def forward(self, *args):
        return self.func(*args)


def destroy_mp():
    # destroy distributed process if already started
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def init_cpu_env():
    """
    If the CPU implementation uses a distributed framework,
    We will need to call this function first.
    """
    destroy_mp()
    print("Initializing cpu env")
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8080"
    os.environ["RANK"] = "0"
    torch.distributed.init_process_group(backend="gloo")
    parallel_state.initialize_model_parallel()


def validate_accuracy(
    neuron_model,
    inputs: List[Tuple],
    expected_outputs: Optional[List] = None,
    cpu_callable: Optional[Callable] = None,
    assert_close_kwargs: Dict = {},
):
    """
    Validates the accuracy of a Neuron model. This function tests that the model produces expected
    outputs, which you can provide and/or produce on CPU. To compare outputs, this function uses
    torch_neuronx.testing.assert_close. If the output isn't similar, this function raises an
    AssertionError.

    Args:
        neuron_model: The Neuron model to validate.
        inputs: The list of inputs to use to run the model. Each input is passed to the model's
            forward function.
        expected_outputs: The list of expected outputs for each input. If not provided, this
            function compares against the CPU output for each input.
        cpu_callable: The callable to use to produce output on CPU.
        assert_close_kwargs: The kwargs to pass to torch_neuronx.testing.assert_close.
    """
    if expected_outputs is None and cpu_callable is None:
        raise ValueError("Provide expected_outputs or a cpu_callable to produce expected outputs")

    if not _is_tensor_tuple_list(inputs):
        raise ValueError("inputs must be a list of tensor tuples")
    if len(inputs) == 0:
        raise ValueError("inputs must not be empty")

    if expected_outputs is None:
        expected_outputs = [None] * len(inputs)
    if not isinstance(expected_outputs, list):
        raise ValueError("expected_outputs must be a list")
    if len(expected_outputs) != len(inputs):
        raise ValueError("len(expected_outputs) must match len(inputs)")

    for input, expected_output in zip(inputs, expected_outputs):
        logger.info(f"Validating model accuracy with input: {input}")
        if cpu_callable is not None:
            cpu_output = cpu_callable(*input)
            logger.info(f"CPU output: {cpu_output}")
            if expected_output is not None:
                torch_neuronx.testing.assert_close(expected_output, cpu_output, **assert_close_kwargs)
            else:
                expected_output = cpu_output

        neuron_output = neuron_model(*input)
        logger.info(f"Expected output: {expected_output}")
        logger.info(f"Neuron output: {neuron_output}")
        torch_neuronx.testing.assert_close(expected_output, neuron_output, **assert_close_kwargs)
        logger.info(f"Model is accurate for input: {input}")


def build_function(
    func: Callable,
    example_inputs: List[Tuple[torch.Tensor]],
    tp_degree: int = 1,
    compiler_args: Optional[str] = None,
    compiler_workdir: Optional[str] = None,
    priority_model_idx: Optional[int] = 0,
    logical_nc_config: int = 1,
    dry_run: bool = False,
):
    """
    Compiles a function to Neuron.

    If the function has non-tensor inputs, you must convert it to a function that only takes
    tensor inputs. You can use `partial` to do this, where you provide the non-tensor inputs as
    constants in the partial function. This step is necessary because all inputs must be tensors
    in a Neuron model.

    Args:
        func: The function to compile.
        example_inputs: The list of example inputs to use to trace the function. This list must
            contain exactly one tuple of tensors.
        tp_degree: The TP degree to use. Defaults to 1.
        compiler_args: The compiler args to use.
        compiler_workdir: Where to save compiler artifacts. Defaults to a tmp folder with a UUID
            for uniqueness.
        priority_model_idx: default 0 indicating enable WLO (weight layout optimization)
        logical_nc_config: The number of logical neuron cores to use. Defaults to 1.
        dry_run: Whether to stop after trace (before compile). If priority_model_idx is set, then
            dry run mode compiles the priority model in order to produce the weight layout
            optimization model.

    Returns:
        The Neuron model, or None if dry run mode is enabled.
    """
    return build_module(
        module_cls=FunctionModule,
        example_inputs=example_inputs,
        tp_degree=tp_degree,
        compiler_args=compiler_args,
        compiler_workdir=compiler_workdir,
        module_init_kwargs={"func": func},
        priority_model_idx=priority_model_idx,
        logical_nc_config=logical_nc_config,
        dry_run=dry_run,
    )


def build_module(
    module_cls,
    example_inputs: List[Tuple[torch.Tensor]],
    module_init_kwargs: Dict = {},
    tp_degree: int = 1,
    compiler_args: Optional[str] = None,
    compiler_workdir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    priority_model_idx: Optional[int] = 0,
    logical_nc_config: int = 1,
    dry_run: bool = False,
):
    """
    Compiles a module to Neuron.

    Args:
        module_cls: The module class to compile.
        example_inputs: The list of example inputs to use to trace the module. This list must
            contain exactly one tuple of tensors.
        tp_degree: The TP degree to use. Defaults to 1.
        module_init_kwargs: The kwargs to pass when initializing the module.
        compiler_args: The compiler args to use.
        compiler_workdir: Where to save compiler artifacts. Defaults to a tmp folder with a UUID
            for uniqueness.
        checkpoint_path: The path to the checkpoint to load. By default, this function saves the
            module state dict to use as the checkpoint.
        priority_model_idx: default 0 indicating enable WLO (weight layout optimization)
        logical_nc_config: The number of logical neuron cores to use. Defaults to 1.
        dry_run: Whether to stop after trace (before compile). If priority_model_idx is set, then
            dry run mode compiles the priority model in order to produce the weight layout
            optimization model.

    Returns:
        The Neuron model, or None if dry run mode is enabled.
    """
    if not _is_tensor_tuple_list(example_inputs):
        raise ValueError("example_inputs must be a list of tensor tuples")
    if len(example_inputs) != 1:
        # Bucketing isn't currently supported for this utility.
        raise ValueError("example_inputs must contain exactly one input")

    _id = uuid.uuid4()
    test_workdir = Path(f"/tmp/nxdi_test_{_id}")
    compiler_workdir = Path(compiler_workdir) if compiler_workdir is not None else test_workdir / "compiler_workdir"
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else test_workdir / "checkpoint.pt"
    logger.info(f"Saving to compiler workdir: {compiler_workdir}")
    logger.info(f"Using checkpoint path: {checkpoint_path}")

    if not checkpoint_path.exists():
        _save_checkpoint(module_cls, module_init_kwargs, checkpoint_path, tp_degree)

    if not compiler_workdir.exists():
        compiler_workdir.parent.mkdir(parents=True, exist_ok=True)

    model_builder = ModelBuilder(
        router=None,
        tp_degree=tp_degree,
        checkpoint_loader=partial(_load_checkpoint, checkpoint_path),
        compiler_workdir=compiler_workdir,
        logical_nc_config=logical_nc_config,
    )

    module_instance_cls = partial(module_cls, **module_init_kwargs)
    model_builder.add(
        key=_get_module_name(module_cls, module_init_kwargs),
        model_instance=BaseModelInstance(module_instance_cls, input_output_aliases={}),
        example_inputs=example_inputs,
        compiler_args=compiler_args,
        priority_model_idx=priority_model_idx,
    )

    neuron_model = model_builder.trace(initialize_model_weights=True, dry_run=dry_run)
    if not dry_run:
        neuron_model.nxd_model.initialize_with_saved_weights(start_rank_tensor=torch.tensor([0]))
    return neuron_model


def _get_module_name(module_cls, module_init_kwargs):
    if module_cls == FunctionModule:
        module_cls = module_init_kwargs["func"]
    if isinstance(module_cls, partial):
        module_cls = module_cls.func
    return module_cls.__name__


def _save_checkpoint(module_cls, module_init_kwargs, checkpoint_path, tp_degree=1):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Parallel state is required to init modules that have distributed layers like RPL/CPL.
    torch.distributed.init_process_group(backend="xla", rank=0, world_size=tp_degree)
    parallel_state.initialize_model_parallel(tp_degree)

    # Set the parallel state random seed to ensure random weights match modules initialized on CPU.
    set_seed(0)
    module = module_cls(**module_init_kwargs)
    torch.save(module.state_dict(), checkpoint_path)

    destroy_mp()


def _load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def _is_tensor_tuple_list(tensor_tuple_list):
    return isinstance(tensor_tuple_list, list) and all(_is_tensor_tuple(item) for item in tensor_tuple_list)


def _is_tensor_tuple(tensor_tuple):
    return isinstance(tensor_tuple, tuple) and all(isinstance(tensor, torch.Tensor) for tensor in tensor_tuple)
