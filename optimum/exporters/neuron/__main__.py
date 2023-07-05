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

import argparse
import inspect
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Optional, Union

from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig

from ...neuron.utils import is_neuron_available, is_neuronx_available
from ...utils import logging
from ...utils.save_utils import maybe_save_preprocessors
from ..error_utils import AtolError, OutputMatchError, ShapeError
from ..tasks import TasksManager
from .convert import export_models, validate_models_outputs
from .model_configs import *  # noqa: F403
from .utils import (
    build_stable_diffusion_components_mandatory_shapes,
    get_stable_diffusion_models_for_export,
)


if is_neuron_available():
    from ...commands.export.neuron import parse_args_neuron

    NEURON_COMPILER = "Neuron"


if is_neuronx_available():
    from ...commands.export.neuronx import parse_args_neuronx as parse_args_neuron  # noqa: F811

    NEURON_COMPILER = "Neuronx"


logger = logging.get_logger()
logger.setLevel(logging.INFO)


def infer_compiler_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    # infer compiler kwargs
    auto_cast = None if args.auto_cast == "none" else args.auto_cast
    auto_cast_type = None if auto_cast is None else args.auto_cast_type
    compiler_kwargs = {"auto_cast": auto_cast, "auto_cast_type": auto_cast_type}
    if hasattr(args, "disable_fast_relayout"):
        compiler_kwargs["disable_fast_relayout"] = getattr(args, "disable_fast_relayout")
    if hasattr(args, "disable_fallback"):
        compiler_kwargs["disable_fallback"] = getattr(args, "disable_fallback")

    return compiler_kwargs


def infer_task(task: str, model_name_or_path: str) -> str:
    if task == "auto":
        try:
            task = TasksManager.infer_task_from_model(model_name_or_path)
        except KeyError as e:
            raise KeyError(
                "The task could not be automatically inferred. Please provide the argument --task with the task "
                f"from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
        except RequestsConnectionError as e:
            raise RequestsConnectionError(
                f"The task could not be automatically inferred as this is available only for models hosted on the Hugging Face Hub. Please provide the argument --task with the relevant task from {', '.join(TasksManager.get_all_tasks())}. Detailed error: {e}"
            )
    return task


def normalize_input_shapes(task: str, args: argparse.Namespace) -> Dict[str, int]:
    if task == "stable-diffusion":
        mandatory_shapes = {
            name: getattr(args, name, None)
            for name in getattr(inspect.getfullargspec(build_stable_diffusion_components_mandatory_shapes), "args")
        }
        input_shapes = build_stable_diffusion_components_mandatory_shapes(**mandatory_shapes)
    else:
        config = AutoConfig.from_pretrained(args.model)
        model_type = config.model_type.replace("_", "-")
        neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model_type=model_type, exporter="neuron", task=task
        )
        mandatory_axes = neuron_config_constructor.func.get_mandatory_axes_for_task(task)
        input_shapes = {name: getattr(args, name) for name in mandatory_axes}

    return input_shapes


def main_export(
    model_name_or_path: str,
    output: Union[str, Path],
    compiler_kwargs: Dict[str, Any],
    task: str = "auto",
    dynamic_batch_size: bool = False,
    atol: Optional[float] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
    subfolder: str = "",
    revision: str = "main",
    force_download: bool = False,
    local_files_only: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    do_validation: bool = True,
    **input_shapes,
):
    output = Path(output)
    if not output.parent.exists():
        output.parent.mkdir(parents=True)

    model = TasksManager.get_model_from_task(
        task,
        model_name_or_path,
        subfolder=subfolder,
        revision=revision,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token,
        local_files_only=local_files_only,
        force_download=force_download,
        trust_remote_code=trust_remote_code,
        framework="pt",
    )
    configs = {}

    if task != "stable-diffusion":
        neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="neuron", task=task
        )
        if is_neuron_available() and dynamic_batch_size is True and "batch_size" in input_shapes:
            input_shapes["batch_size"] = 1
        neuron_config = neuron_config_constructor(model.config, dynamic_batch_size=dynamic_batch_size, **input_shapes)
        if atol is None:
            atol = neuron_config.ATOL_FOR_VALIDATION
        output_model_names = ["model.neuron"]
        models_and_neuron_configs = {"model": (model, neuron_config)}
        maybe_save_preprocessors(model, output.parent)

    if task == "stable-diffusion":
        if not is_neuronx_available():
            raise RuntimeError("Stable diffusion needs neuronx-cc support which is not installed. ")
        output_model_names = [
            "text_encoder/model.neuron",
            "vae/decoder.neuron",
            "unet/model.neuron",
            "vae/post_quant_conv.neuron",
        ]
        models_and_neuron_configs = get_stable_diffusion_models_for_export(
            model,
            dynamic_batch_size=dynamic_batch_size,
            **input_shapes,
        )
        configs["vae_decoder"] = configs["vae_conv"] = model.vae.config
        # Saving the model config and preprocessor as this is needed sometimes.
        model.tokenizer.save_pretrained(output.joinpath("tokenizer"))
        model.scheduler.save_pretrained(output.joinpath("scheduler"))
        if model.feature_extractor is not None:
            model.feature_extractor.save_pretrained(output.joinpath("feature_extractor"))
        # Save SD pipeline model index
        model.save_config(output)

    neuron_inputs, neuron_outputs = export_models(
        models_and_neuron_configs=models_and_neuron_configs,
        output_dir=output,
        output_file_names=output_model_names,
        compiler_kwargs=compiler_kwargs,
        configs=configs,
    )

    del model

    # Validate compiled model
    if do_validation is True:
        try:
            validate_models_outputs(
                models_and_neuron_configs=models_and_neuron_configs,
                neuron_named_outputs=neuron_outputs,
                output_dir=output,
                atol=atol,
                neuron_files_subpaths=output_model_names,
            )

            logger.info(
                f"The {NEURON_COMPILER} export succeeded and the exported model was saved at: "
                f"{output.parent.as_posix()}"
            )
        except ShapeError as e:
            raise e
        except AtolError as e:
            logger.warning(
                f"The {NEURON_COMPILER} export succeeded with the warning: {e}.\n The exported model was saved at: "
                f"{output.parent.as_posix()}"
            )
        except OutputMatchError as e:
            logger.warning(
                f"The {NEURON_COMPILER} export succeeded with the warning: {e}.\n The exported model was saved at: "
                f"{output.parent.as_posix()}"
            )
        except Exception as e:
            logger.error(
                f"An error occured with the error message: {e}.\n The exported model was saved at: "
                f"{output.parent.as_posix()}"
            )


def main():
    parser = ArgumentParser(f"Hugging Face Optimum {NEURON_COMPILER} exporter")

    parse_args_neuron(parser)

    # Retrieve CLI arguments
    args = parser.parse_args()

    task = infer_task(args.task, args.model)
    compiler_kwargs = infer_compiler_kwargs(args)
    input_shapes = normalize_input_shapes(task, args)

    main_export(
        model_name_or_path=args.model,
        output=args.output,
        compiler_kwargs=compiler_kwargs,
        task=task,
        dynamic_batch_size=args.dynamic_batch_size,
        atol=args.atol,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        **input_shapes,
    )


if __name__ == "__main__":
    main()
