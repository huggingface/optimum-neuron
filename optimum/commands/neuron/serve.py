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
"""Defines the command to serve a Neuron model."""

import os
import warnings
from argparse import ArgumentParser

from optimum.commands.base import BaseOptimumCLICommand
from optimum.utils import logging
from transformers import AutoConfig

from ...neuron.cache.hub_cache import select_hub_cached_entries
from ...neuron.configuration_utils import NeuronConfig
from ...neuron.utils import DTYPE_MAPPER
from ...neuron.utils.import_utils import is_vllm_available
from ...neuron.utils.instance import current_instance_type
from ...neuron.utils.require_utils import requires_torch_neuronx, requires_vllm
from ...neuron.utils.system import get_available_cores


if is_vllm_available():
    import asyncio

    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils import FlexibleArgumentParser


logger = logging.get_logger()


class ServeCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        # The optimum-cli uses a strict parsing and does not allow unknown args.
        # However, parser.parse_args still internally calls parser.parse_known_args.
        # We therefore patch parser.parse_known_args to avoid returning the unknown
        # args for this command, and store them in the args instead.
        parser._original_parse_known_args = parser.parse_known_args

        def parse_known_args(args, namespace):
            args, unknown_args = parser._original_parse_known_args(args, namespace)
            if hasattr(args, "func") and type(args.func(args)) is ServeCommand:
                # Just for this command, instead of returning the unknown args, we store them
                args._unknown_args = unknown_args
                return args, None
            return args, unknown_args

        parser.parse_known_args = parse_known_args
        # Add arguments that are explicitly used by the run command
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            required=True,
            help="Model ID on huggingface.co or path on disk to load model from.",
        )
        parser.add_argument(
            "--served_model_name",
            "--served-model-name",
            type=str,
            default=None,
            help="The model name(s) used in the API. If not specified, the model name will be the same as the `--model` argument.",
        )
        parser.add_argument(
            "--tensor_parallel_size",
            "--tensor-parallel-size",
            type=int,
            help="Tensor parallelism size, the number of neuron cores on which to shard the model.",
        )
        parser.add_argument(
            "--batch_size",
            "--batch-size",
            type=int,
            help="The maximum batch size used when serving the model.",
        )
        parser.add_argument(
            "--sequence_length",
            "--sequence-length",
            type=int,
            help="The sequence length used when serving the model.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8080,
            help="The port on which to serve the model.",
        )
        parser.add_argument(
            "--allow_non_cached_model",
            "--allow-non-cached-model",
            action="store_true",
            default=False,
            help="If set, export the model even if no cached configuration exists.",
        )

    @requires_vllm
    @requires_torch_neuronx
    def run(self):
        model_name_or_path = self.args.model
        model_id = self.args.served_model_name
        if model_id is None:
            model_id = model_name_or_path
        else:
            if model_name_or_path != model_id:
                logger.info(f"Serving model {model_id} at path {model_name_or_path}")
        instance_type = current_instance_type()
        batch_size = self.args.batch_size
        sequence_length = self.args.sequence_length
        tensor_parallel_size = self.args.tensor_parallel_size
        config = AutoConfig.from_pretrained(model_name_or_path)
        torch_dtype = DTYPE_MAPPER.pt(config.dtype)
        try:
            # Look for a NeuronConfig in the model directory
            neuron_config = NeuronConfig.from_pretrained(model_name_or_path)
        except EnvironmentError:
            neuron_config = None
        if neuron_config is not None:
            # This is a Neuron model: retrieve and check the export arguments
            if neuron_config.target != instance_type:
                raise ValueError(
                    f"The neuron model is compiled for {neuron_config.target} and cannot run on a {instance_type} instance."
                )
            if batch_size is None:
                batch_size = neuron_config.batch_size
            elif batch_size != neuron_config.batch_size:
                raise ValueError(
                    f"The specified batch_size {batch_size} is inconsistent"
                    f"with the one used to export the neuron model ({neuron_config.batch_size})"
                )
            if sequence_length is None:
                sequence_length = neuron_config.sequence_length
            elif sequence_length != neuron_config.sequence_length:
                raise ValueError(
                    f"The specified sequence length {sequence_length} is inconsistent"
                    f"with the one used to export the neuron model ({neuron_config.sequence_length})"
                )
            if tensor_parallel_size is None:
                tensor_parallel_size = neuron_config.tp_degree
            elif tensor_parallel_size != neuron_config.tp_degree:
                raise ValueError(
                    f"The specified tensor parallel size {tensor_parallel_size} is inconsistent"
                    f"with the one used to export the neuron model ({neuron_config.tp_degree})"
                )
            logger.info(f"Loading Neuron model: {model_name_or_path}")
        else:
            # Model needs to be exported: look for compatible hub cached configs

            cached_entries = select_hub_cached_entries(
                model_name_or_path,
                task="text-generation",
                instance_type=instance_type,
                batch_size=batch_size,
                sequence_length=sequence_length,
                tensor_parallel_size=tensor_parallel_size,
                torch_dtype=torch_dtype,
            )
            # Filter out entries that do not fit on the target host
            available_cores = get_available_cores()
            filtered_entries = [e for e in cached_entries if e["tp_degree"] <= available_cores]
            if len(filtered_entries) == 0:
                if self.args.allow_non_cached_model:
                    warning_msg = f"{model_id} is not a neuron model, and no cached configuration is available using"
                    warning_msg += f" instance type {instance_type},"
                    if batch_size is None:
                        batch_size = 1
                        warning_msg += " default"
                    warning_msg += f" batch size = {batch_size},"
                    if sequence_length is None:
                        sequence_length = 2048
                        warning_msg += " default"
                    warning_msg += f" sequence length = {sequence_length},"
                    if tensor_parallel_size is None:
                        tensor_parallel_size = available_cores
                        warning_msg += " default"
                    warning_msg += f" tp = {tensor_parallel_size},"
                    if torch_dtype is None:
                        torch_dtype = DTYPE_MAPPER.pt("bfloat16")
                        warning_msg += " default"
                    warning_msg += f" dtype = {torch_dtype}."
                    warning_msg += " The compilation might fail or the model might not fit on the target instance."
                    warnings.warn(warning_msg)
                else:
                    hub_cache_url = "https://huggingface.co/aws-neuron/optimum-neuron-cache"  # noqa: E501
                    neuron_export_url = "https://huggingface.co/docs/optimum-neuron/main/en/guides/export_model"  # noqa: E501
                    error_msg = f"No cached version found for {model_id}"
                    error_msg += f" instance type {instance_type},"
                    if batch_size is not None:
                        error_msg += f", batch size = {batch_size}"
                    if sequence_length is not None:
                        error_msg += f", sequence length = {sequence_length},"
                    if tensor_parallel_size is not None:
                        error_msg += f", tp = {tensor_parallel_size}"
                    if torch_dtype is not None:
                        error_msg += f", dtype = {torch_dtype}"
                    error_msg += (
                        f".You can start a discussion to request it on {hub_cache_url} "
                        "Alternatively, you can export your own neuron model "
                        f"as explained in {neuron_export_url}"
                    )
                    raise ValueError(error_msg)
            else:
                # Sort entries by decreasing tensor parallel size, batch size, sequence length
                filtered_entries = sorted(
                    filtered_entries,
                    key=lambda x: (x["tp_degree"], x["batch_size"], x["sequence_length"]),
                    reverse=True,
                )
                # Export the model with the best matching configuration
                selected_entry = filtered_entries[0]
                batch_size = selected_entry["batch_size"]
                sequence_length = selected_entry["sequence_length"]
                tensor_parallel_size = selected_entry["tp_degree"]
                torch_dtype = DTYPE_MAPPER.pt(selected_entry["torch_dtype"])
                warning_msg = f"{model_id} is not a neuron model, but a cached configuration is available using"
                warning_msg += f" instance type {instance_type},"
                warning_msg += f" batch size = {batch_size},"
                warning_msg += f" sequence length = {sequence_length},"
                warning_msg += f" tp = {tensor_parallel_size},"
                warning_msg += f" dtype = {torch_dtype}."
                logger.warning(warning_msg)

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        vllm_parser = make_arg_parser(FlexibleArgumentParser())
        command = [
            "--model",
            self.args.model,
            "--served_model_name",
            model_id,
            "--port",
            str(self.args.port),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--max-num-seqs",
            str(batch_size),
            "--max-model-len",
            str(sequence_length),
            "--dtype",
            str(torch_dtype).split(".")[-1],
        ]
        if self.args.allow_non_cached_model:
            command.append("--model-loader-extra-config=allow_non_cached_model")
        if hasattr(self.args, "_unknown_args"):
            command.extend(self.args._unknown_args)
        vllm_args = vllm_parser.parse_args(command)
        validate_parsed_serve_args(vllm_args)

        asyncio.run(run_server(vllm_args))
