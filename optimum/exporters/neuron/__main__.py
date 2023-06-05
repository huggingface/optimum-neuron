# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Entry point to the optimum.exporters.neuron command line."""

import copy
from argparse import ArgumentParser

from ...neuron.utils import is_neuron_available, is_neuronx_available, store_compilation_config
from ...utils import logging
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .convert import export, validate_model_outputs
from .model_configs import *  # noqa: F403


if is_neuron_available():
    from ...commands.export.neuron import parse_args_neuron

    NEURON_COMPILER = "Neuron"

if is_neuronx_available():
    from ...commands.export.neuronx import parse_args_neuronx as parse_args_neuron  # noqa: F811

    NEURON_COMPILER = "Neuronx"


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def main():
    parser = ArgumentParser(f"Hugging Face Optimum {NEURON_COMPILER} exporter")

    parse_args_neuron(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output.joinpath("model.neuron")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Infer the task
    task = args.task
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(args.model)
        except KeyError as e:
            raise KeyError(
                "The task could not be automatically inferred. Please provide the argument --task with the task "
                f"from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )

    model = TasksManager.get_model_from_task(
        task, args.model, framework="pt", cache_dir=args.cache_dir, trust_remote_code=args.trust_remote_code
    )
    ref_model = copy.deepcopy(model)

    neuron_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="neuron", task=task)
    # TODO: find a cleaner way to do this.
    input_shapes = {
        name: getattr(args, name) for name in neuron_config_constructor.func.get_mandatory_axes_for_task(task)
    }
    if is_neuron_available() and args.dynamic_batch_size is True and "batch_size" in input_shapes:
        input_shapes["batch_size"] = 1
    neuron_config = neuron_config_constructor(model.config, dynamic_batch_size=args.dynamic_batch_size, **input_shapes)

    if args.atol is None:
        args.atol = neuron_config.ATOL_FOR_VALIDATION

    # Get compilation arguments
    auto_cast = None if args.auto_cast == "none" else args.auto_cast
    auto_cast_type = None if auto_cast is None else args.auto_cast_type
    compiler_kwargs = {"auto_cast": auto_cast, "auto_cast_type": auto_cast_type}
    if hasattr(args, "disable_fast_relayout"):
        compiler_kwargs["disable_fast_relayout"] = getattr(args, "disable_fast_relayout")
    if hasattr(args, "disable_fallback"):
        compiler_kwargs["disable_fallback"] = getattr(args, "disable_fallback")

    neuron_inputs, neuron_outputs = export(
        model=model,
        config=neuron_config,
        output=args.output,
        **compiler_kwargs,
    )

    # For torch_neuron, batch_size must be equal to 1 when dynamic batching is on.
    store_compilation_config(
        model.config, input_shapes, compiler_kwargs, neuron_inputs, neuron_outputs, args.dynamic_batch_size
    )

    # Saving the model config and preprocessor as this is needed sometimes.
    model.config.save_pretrained(args.output.parent)
    maybe_save_preprocessors(args.model, args.output.parent)

    # Validate compiled model
    try:
        validate_model_outputs(
            config=neuron_config,
            reference_model=ref_model,
            neuron_model_path=args.output,
            neuron_named_outputs=neuron_config.outputs,
            atol=args.atol,
        )

        logger.info(
            f"The {NEURON_COMPILER} export succeeded and the exported model was saved at: "
            f"{args.output.parent.as_posix()}"
        )
    except ShapeError as e:
        raise e
    except AtolError as e:
        logger.warning(
            f"The {NEURON_COMPILER} export succeeded with the warning: {e}.\n The exported model was saved at: "
            f"{args.output.parent.as_posix()}"
        )
    except OutputMatchError as e:
        logger.warning(
            f"The {NEURON_COMPILER} export succeeded with the warning: {e}.\n The exported model was saved at: "
            f"{args.output.parent.as_posix()}"
        )
    except Exception as e:
        logger.error(
            f"An error occured with the error message: {e}.\n The exported model was saved at: "
            f"{args.output.parent.as_posix()}"
        )


if __name__ == "__main__":
    main()
