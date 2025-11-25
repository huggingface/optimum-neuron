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
"""Script to cache LLM models for inference."""

import argparse
import json
import logging
import subprocess
import tempfile
import time

import requests
from optimum.exporters.tasks import TasksManager

from optimum.neuron.utils.instance import SUPPORTED_INSTANCE_TYPES


# Example usage:
# huggingface-cli login --token hf_xxx # access to cache repo
# python tools/auto_fill_llm_cache.py --model_id "HuggingFaceH4/zephyr-7b-beta" --batch_size 1 --sequence_length 2048 --tensor_parallel_size 2
# Alternative provide json config file as local file or remote file (https://) with the following format
# {
#    "meta-llama/Llama-2-7b-chat-hf": [
#        {  "batch_size": 1, "sequence_length": 4096, "tensor_parallel_size": 2, "instance_type": "trn1" },
#        {  "batch_size": 1, "sequence_length": 4096, "tensor_parallel_size": 4, "instance_type": "trn2" }
#    ]
# }
# Local file Example usage:
# python tools/auto_fill_llm_cache.py --config_file test.json
# Remote file Example usage:
# python tools/auto_fill_llm_cache.py --config_file https://huggingface.co/aws-neuron/optimum-neuron-cache/raw/main/inference-cache-config/gpt2.json

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()


def build_llm_command(
    model_id: str,
    instance_type: str,
    batch_size: int,
    sequence_length: int,
    tensor_parallel_size: int,
    task: str,
    output_dir: str,
):
    compile_command = [
        "optimum-cli",
        "export",
        "neuron",
        "-m",
        model_id,
        "--instance_type",
        instance_type,
        "--batch_size",
        str(batch_size),
        "--sequence_length",
        str(sequence_length),
        "--tensor_parallel_size",
        str(tensor_parallel_size),
        "--task",
        task,
        output_dir,
    ]
    return compile_command


def compile_and_cache_model(
    model_id: str,
    instance_type: str,
    batch_size: int,
    sequence_length: int,
    tensor_parallel_size: int,
    task: str | None = None,
):
    start = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        if task is None:
            task = TasksManager.infer_task_from_model(model_id)
        # Compile model with Optimum for specific configurations
        compile_command = build_llm_command(
            model_id,
            instance_type=instance_type,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
            task=task,
            output_dir=temp_dir,
        )
        logger.info(f"Running compile command: {' '.join(compile_command)}")
        try:
            subprocess.run(compile_command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile model: {e}")
            return

        # Synchronize compiled model to Hugging Face Hub
        cache_sync_command = ["optimum-cli", "neuron", "cache", "synchronize"]
        logger.info(f"Running cache synchronize command: {' '.join(cache_sync_command)}")
        subprocess.run(cache_sync_command, check=True)

    # Log time taken
    logger.info(f"Compiled and cached model {model_id} w{time.time() - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and cache a model to the Hugging Face Hub.")
    parser.add_argument("--model_id", type=str, help="Hugging Face model ID to compile.")
    parser.add_argument("--task", type=str, help="Task for compilation (mandatory for encoders).")
    parser.add_argument(
        "--instance_type",
        type=str,
        choices=SUPPORTED_INSTANCE_TYPES,
        default=SUPPORTED_INSTANCE_TYPES[0],
        help="The target instance type for compilation.",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size for compilation.")
    parser.add_argument("--sequence_length", type=int, help="Sequence length for compilation.")
    parser.add_argument(
        "--tensor_parallel_size", type=int, help="Number of cores on which the model is split for compilation."
    )
    parser.add_argument("--config_file", type=str, help="Path to a json config file with model configurations.")
    args = parser.parse_args()

    # If a config file is provided, compile and cache all models in the file
    if args.config_file:
        logger.info(f"Compiling and caching models from config file: {args.config_file}")
        # check if config file starts with https://
        if args.config_file.startswith("https://"):
            response = requests.get(args.config_file)
            response.raise_for_status()
            config = response.json()
        else:
            with open(args.config_file, "r") as f:
                config = json.load(f)
        for model_id, configs in config.items():
            for model_config in configs:
                compile_and_cache_model(
                    model_id=model_id,
                    instance_type=model_config.get("instance_type", SUPPORTED_INSTANCE_TYPES[0]),
                    batch_size=model_config["batch_size"],
                    sequence_length=model_config.get("sequence_length", None),
                    tensor_parallel_size=model_config.get("tensor_parallel_size", model_config.get("num_cores", None)),
                    task=model_config.get("task", None),
                )
    elif args.model_id is None:
        raise ValueError("You must provide --model_id to compile a model without a config file.")
    else:
        compile_and_cache_model(
            model_id=args.model_id,
            instance_type=args.instance_type,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            tensor_parallel_size=args.tensor_parallel_size,
            task=args.task,
        )
