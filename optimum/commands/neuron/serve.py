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
from argparse import ArgumentParser

from ...neuron.configuration_utils import NeuronConfig
from ...neuron.utils import DTYPE_MAPPER
from ...neuron.utils.import_utils import is_vllm_available
from ...neuron.utils.require_utils import requires_torch_neuronx, requires_vllm
from ...utils import logging
from ..base import BaseOptimumCLICommand


if is_vllm_available():
    import asyncio

    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils import FlexibleArgumentParser


logger = logging.get_logger()


class ServeCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            required=True,
            help="Model ID on huggingface.co or path on disk to load model from.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            choices=["bfloat16", "float16"],
            help="Override the default `torch.dtype` and load the model under this dtype. If `None` is passed, the dtype will be automatically derived from the model's weights.",
        )
        parser.add_argument(
            "--tensor_parallel_size",
            type=int,
            help="Tensor parallelism size, the number of neuron cores on which to shard the model.",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            help="The maximum batch size used when serving the model.",
        )
        parser.add_argument(
            "--sequence_length",
            type=int,
            help="The sequence length used when serving the model.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8080,
            help="The port on which to serve the model.",
        )

    @requires_vllm
    @requires_torch_neuronx
    def run(self):
        batch_size = self.args.batch_size
        sequence_length = self.args.sequence_length
        tensor_parallel_size = self.args.tensor_parallel_size
        torch_dtype = None if self.args.dtype is None else DTYPE_MAPPER.pt(self.args.dtype)
        if os.path.isdir(self.args.model):
            # The model is a local path, we assume the user has already converted it to a Neuron model
            neuron_config = NeuronConfig.from_pretrained(self.args.model)
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
            if torch_dtype is None:
                torch_dtype = neuron_config.torch_dtype
            elif torch_dtype != neuron_config.torch_dtype:
                raise ValueError(
                    f"The specified dtype {torch_dtype} is inconsistent"
                    f"with the one used to export the neuron model ({neuron_config.torch_dtype})"
                )
            logger.info(f"Loading Neuron model from local path: {self.args.model}")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        vllm_parser = make_arg_parser(FlexibleArgumentParser())
        command = [
            "--model",
            self.args.model,
            "--port",
            str(self.args.port),
        ]
        if tensor_parallel_size is not None:
            command += ["--tensor-parallel-size", str(tensor_parallel_size)]
        if batch_size is not None:
            command += ["--max-num-seqs", str(batch_size)]
        if sequence_length is not None:
            command += ["--max-model-len", str(sequence_length)]
        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                # vLLM does not accept string dtype, convert to torch.dtype
                torch_dtype = DTYPE_MAPPER.vllm(torch_dtype)
            command += ["--dtype", str(torch_dtype).split(".")[-1]]
        vllm_args = vllm_parser.parse_args(command)
        validate_parsed_serve_args(vllm_args)

        asyncio.run(run_server(vllm_args))
