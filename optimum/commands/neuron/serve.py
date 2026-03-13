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
import signal
import sys
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

    from ...neuron.vllm.model_loader import VLLM_2_TRANSFORMERS_TASK_MAPPING
    from ...neuron.vllm.reverse_proxy import RoundRobinProxy
    from ...neuron.vllm.server_manager import VLLMServerManager


logger = logging.get_logger()


def _allocate_internal_ports(user_port: int, count: int) -> list[int]:
    """Allocate *count* internal ports starting right after *user_port*.

    Raises ValueError if any port would exceed 65535.
    """
    ports = [user_port + 1 + i for i in range(count)]
    if ports[-1] > 65535:
        raise ValueError(
            f"Internal ports {ports[0]}-{ports[-1]} exceed the valid range "
            f"(user port {user_port} + {count} replicas). "
            f"Use a lower --port value."
        )
    return ports


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
            "--task",
            type=str,
            choices=["generate", "embed"],
            default="generate",
            help="The task for which the model is being served.",
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
            "--data-parallel-size",
            type=int,
            default=1,
            help="Number of data-parallel replicas. Each replica uses tensor_parallel_size cores.",
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
        if not is_vllm_available():
            raise ImportError("vLLM is not installed. Please install vLLM to use the serve command.")
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
        data_parallel_size = self.args.data_parallel_size
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
                task=VLLM_2_TRANSFORMERS_TASK_MAPPING[self.args.task],
                instance_type=instance_type,
                batch_size=batch_size,
                sequence_length=sequence_length,
                tensor_parallel_size=tensor_parallel_size,
                torch_dtype=torch_dtype,
            )
            # Filter out entries that do not fit on the target host
            available_cores = get_available_cores()
            filtered_entries = [e for e in cached_entries if e["tp_degree"] * data_parallel_size <= available_cores]
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

        if data_parallel_size > 1:
            available_cores = get_available_cores()
            total_cores_needed = tensor_parallel_size * data_parallel_size
            if total_cores_needed > available_cores:
                raise ValueError(
                    f"Data parallelism requires {total_cores_needed} cores "
                    f"({tensor_parallel_size} TP x {data_parallel_size} DP) but only "
                    f"{available_cores} cores are available."
                )

        # Build the vLLM command arguments.
        vllm_command = [
            "--model",
            self.args.model,
            "--served_model_name",
            model_id,
            "--task",
            self.args.task,
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
            vllm_command.append("--model-loader-extra-config=allow_non_cached_model")
        if hasattr(self.args, "_unknown_args"):
            vllm_command.extend(self.args._unknown_args)

        if data_parallel_size == 1:
            # Single server: run directly in this process.
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            vllm_parser = make_arg_parser(FlexibleArgumentParser())
            full_command = ["--port", str(self.args.port)] + vllm_command
            vllm_args = vllm_parser.parse_args(full_command)
            validate_parsed_serve_args(vllm_args)
            asyncio.run(run_server(vllm_args))
        else:
            # Data-parallel: spawn N independent vLLM server processes behind
            # a round-robin reverse proxy, each with its own NeuronCores.
            #
            # Why external LB instead of vLLM's built-in --data-parallel-size:
            # vLLM's internal DP mode routes all N engine cores through a single
            # API server process that handles tokenization, detokenization, and
            # SSE streaming for every connection. Under high concurrency this
            # single process becomes the bottleneck and throughput does not
            # scale linearly with the number of DP replicas. The vLLM
            # --api-server-count option (multiple API server processes sharing
            # the same engine cores) does not help either, as the bottleneck
            # is in the ZMQ transport between API servers and engine cores.
            # Spawning N fully independent servers with a reverse proxy in
            # front achieves the expected linear throughput scaling.
            user_port = self.args.port
            internal_ports = _allocate_internal_ports(user_port, data_parallel_size)

            manager = VLLMServerManager(
                ports=internal_ports,
                tp_size=tensor_parallel_size,
                vllm_args=vllm_command,
            )
            proxy = RoundRobinProxy(upstream_ports=internal_ports, listen_port=user_port)

            async def run_data_parallel():
                # Install a SIGTERM handler that cancels the current task so the
                # finally block runs and manager.shutdown() cleans up children.
                main_task = asyncio.current_task()
                loop = asyncio.get_running_loop()
                loop.add_signal_handler(signal.SIGTERM, main_task.cancel)

                manager.start()
                try:
                    # Start monitoring immediately so a crashed backend
                    # surfaces right away instead of waiting 600s.
                    monitor_task = asyncio.create_task(asyncio.to_thread(manager.monitor))
                    wait_task = asyncio.create_task(proxy.wait_for_backends(timeout=600))

                    # If a backend crashes during startup the monitor task
                    # finishes first, letting us fail fast.
                    done, pending = await asyncio.wait(
                        [monitor_task, wait_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Only cancel wait_task — monitor_task must stay alive
                    # for the proxy phase below.
                    if wait_task in pending:
                        wait_task.cancel()

                    # If the monitor finished first, a backend died.
                    if monitor_task in done:
                        ret = monitor_task.result()
                        if ret is not None:
                            logger.error("A vLLM server exited with code %d during startup", ret)
                            sys.exit(ret or 1)

                    logger.info(
                        "All %d vLLM servers ready, proxy listening on port %d",
                        data_parallel_size,
                        user_port,
                    )

                    proxy_task = asyncio.create_task(proxy.run())
                    done, pending = await asyncio.wait(
                        [proxy_task, monitor_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()

                    # Propagate non-zero exit code from a crashed backend.
                    if monitor_task in done:
                        ret = monitor_task.result()
                        if ret:
                            logger.error("A vLLM server exited with code %d", ret)
                            sys.exit(ret)
                finally:
                    manager.shutdown()

            asyncio.run(run_data_parallel())
