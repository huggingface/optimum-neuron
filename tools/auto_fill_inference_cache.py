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
"""Script to cache models for inference."""
import json
import logging
import os
import subprocess
import argparse
import tempfile
import time
from huggingface_hub import login
from optimum.neuron import version as optimum_neuron_version
import re

# Example usage:
# huggingface-cli login --token hf_xxx # access to cache repo
# python tools/cache_model_for_inference.py --hf_model_id "HuggingFaceH4/zephyr-7b-beta" --batch_size 1 --sequence_length 2048 --num_cores 2 --auto_cast_type fp16
# Alternative provide json config file with the following format:
# python tools/cache_model_for_inference.py --config_file test.json
# {
#    "meta-llama/Llama-2-7b-chat-hf": [
#        {  "batch_size": 1, "sequence_length": 2048, "num_cores": 2, "auto_cast_type": "fp16" },
#        {  "batch_size": 2, "sequence_length": 2048, "num_cores": 2, "auto_cast_type": "bf16" }
#    ]
# }

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()


def get_neuronx_cc_version():
    result = subprocess.run(["neuronx-cc", "--version"], capture_output=True, text=True)
    version_match = re.search(r"NeuronX Compiler version ([\d\.]+\+[a-f0-9]+)", result.stderr)
    if version_match:
        return version_match.group(1)
    else:
        raise ValueError("Version information not found in the output")


def get_aws_neuronx_tools_version():
    output = subprocess.check_output(["apt", "show", "aws-neuronx-tools"], text=True)
    version_match = re.search(r"Version: ([\d\.]+)", output)

    if version_match:
        # extract the version number and remove the last two characters (not tracked in optimum)
        return version_match.group(1)[:-2]
    else:
        raise ValueError("Version information not found in the output")


def compile_and_cache_model(hf_model_id, batch_size, sequence_length, num_cores, auto_cast_type):
    start = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Compile model with Optimum for specific configurations
        compile_command = [
            "optimum-cli",
            "export",
            "neuron",
            "-m",
            hf_model_id,
            "--batch_size",
            str(batch_size),
            "--sequence_length",
            str(sequence_length),
            "--num_cores",
            str(num_cores),
            "--auto_cast_type",
            auto_cast_type,
            temp_dir,
        ]
        logger.info(f"Running compile command: {' '.join(compile_command)}")
        try:
            subprocess.run(compile_command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile model: {e}")
            return

        # Synchronize compiled model to Hugging Face Hub
        cache_sync_command = ["optimum-cli", "neuron", "cache", "synchronize"]
        logger.info(f"Running cache synchronize command: {' '.join(cache_sync_command)}")

        try:
            subprocess.run(cache_sync_command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to synchronize compiled model: {e}")
            return

    # Log time taken
    logger.info(f"Compiled and cached model {hf_model_id} w{time.time() - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and cache a model to the Hugging Face Hub.")
    parser.add_argument("--hf_model_id", type=str, help="Hugging Face model ID to compile.")
    parser.add_argument("--batch_size", type=int, help="Batch size for compilation.")
    parser.add_argument("--sequence_length", type=int, help="Sequence length for compilation.")
    parser.add_argument("--num_cores", type=int, help="Number of cores for compilation.")
    parser.add_argument("--auto_cast_type", type=str, choices=["bf16", "fp16"], help="Auto cast type for compilation.")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for authentication if not logged in.")
    parser.add_argument("--config_file", type=str, help="Path to a json config file with model configurations.")
    args = parser.parse_args()

    # Ensure either HF token is provided or user is already logged in
    if args.hf_token:
        logger.info(f"Logging in to Hugging Face Hub with {args.hf_token[:10]}...")
        login(args.hf_token)
    else:
        logger.info("Trying to use existing Hugging Face Hub login or environment variable HF_TOKEN")

    # check and get neuronx-cc version
    neuronx_cc_version = get_neuronx_cc_version()
    sdk_version = get_aws_neuronx_tools_version()
    logger.info(f"Compiler version: {neuronx_cc_version}")
    logger.info(f"Neuron SDK version: {sdk_version}")
    logger.info(f"Optimum Neuron version: {optimum_neuron_version.__version__}")
    logger.info(f"Compatible Optimum Neuron SDK version: {optimum_neuron_version.__sdk_version__} == {sdk_version}")
    assert (
        optimum_neuron_version.__sdk_version__ == sdk_version
    ), f"Optimum Neuron SDK version {optimum_neuron_version.__sdk_version__} is not compatible with installed Neuron SDK version {sdk_version}"

    # If a config file is provided, compile and cache all models in the file
    if args.config_file:
        logger.info(f"Compiling and caching models from config file: {args.config_file}")
        with open(args.config_file, "r") as f:
            config = json.load(f)
        for model_id, configs in config.items():
            for model_config in configs:

                compile_and_cache_model(
                    hf_model_id=model_id,
                    batch_size=model_config["batch_size"],
                    sequence_length=model_config["sequence_length"],
                    num_cores=model_config["num_cores"],
                    auto_cast_type=model_config["auto_cast_type"],
                )
    # Check if all arguments are provided if a config file is not used
    if (
        args.hf_model_id is None
        or args.batch_size is None
        or args.sequence_length is None
        or args.num_cores is None
        or args.auto_cast_type is None
    ):
        raise ValueError(
            "You must provide a --hf_model_id, --batch_size, --sequence_length, --num_cores, and --auto_cast_type to compile a model without a config file."
        )

    # Otherwise, compile and cache a single model
    compile_and_cache_model(
        hf_model_id=args.hf_model_id,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_cores=args.num_cores,
        auto_cast_type=args.auto_cast_type,
    )
