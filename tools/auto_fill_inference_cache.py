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

import argparse
import json
import logging
import re
import subprocess
import tempfile
import time

import huggingface_hub
import requests
from huggingface_hub import get_token, login, logout

from optimum.exporters import TasksManager
from optimum.neuron import version as optimum_neuron_version
from optimum.neuron.utils.version_utils import get_neuronxcc_version


# Example usage:
# huggingface-cli login --token hf_xxx # access to cache repo
# python tools/auto_fill_inference_cache.py --hf_model_id "HuggingFaceH4/zephyr-7b-beta" --batch_size 1 --sequence_length 2048 --num_cores 2 --auto_cast_type fp16
# Alternative provide json config file as local file or remote file (https://) with the following formwat
# {
#    "meta-llama/Llama-2-7b-chat-hf": [
#        {  "batch_size": 1, "sequence_length": 2048, "num_cores": 2, "auto_cast_type": "fp16" },
#        {  "batch_size": 2, "sequence_length": 2048, "num_cores": 2, "auto_cast_type": "bf16" }
#    ]
# }
# Local file Example usage:
# python tools/auto_fill_inference_cache.py --config_file test.json
# Remote file Example usage:
# python tools/auto_fill_inference_cache.py --config_file https://huggingface.co/aws-neuron/optimum-neuron-cache/raw/main/inference-cache-config/gpt2.json

# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()


def get_aws_neuronx_tools_version():
    output = subprocess.check_output(["apt", "show", "aws-neuronx-tools"], text=True)
    version_match = re.search(r"Version: ([\d\.]+)", output)

    if version_match:
        # extract the version number and remove the last two characters (not tracked in optimum)
        return version_match.group(1)[:-2]
    else:
        raise ValueError("Version information not found in the output")


def build_decoder_command(hf_model_id, batch_size, sequence_length, num_cores, auto_cast_type, output_dir):
    if None in [batch_size, sequence_length, num_cores, auto_cast_type]:
        raise ValueError(
            "You must provide --batch_size, --sequence_length, --num_cores and --auto_cast_type for compiling decoder models."
        )
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
        "--task",
        "text-generation",
        output_dir,
    ]
    return compile_command


def build_encoder_command(hf_model_id, task, batch_size, sequence_length, auto_cast, auto_cast_type, output_dir):
    if None in [task, batch_size, sequence_length, auto_cast, auto_cast_type]:
        raise ValueError(
            "You must provide --task, --batch_size, --sequence_length, --auto_cast and --auto_cast_type for compiling encoder models."
        )
    compile_command = [
        "optimum-cli",
        "export",
        "neuron",
        "-m",
        hf_model_id,
        "--task",
        task,
        "--batch_size",
        str(batch_size),
        "--sequence_length",
        str(sequence_length),
        "--auto_cast",
        auto_cast,
        "--auto_cast_type",
        auto_cast_type,
        output_dir,
    ]
    return compile_command


def build_stable_diffusion_command(
    hf_model_id, batch_size, height, width, num_images_per_prompt, auto_cast, auto_cast_type, output_dir
):
    if None in [batch_size, height, width, auto_cast, auto_cast_type]:
        raise ValueError(
            "You must provide --batch_size, --height, --width, --auto_cast and --auto_cast_type for compiling stable diffusion models."
        )
    compile_command = [
        "optimum-cli",
        "export",
        "neuron",
        "-m",
        hf_model_id,
        "--batch_size",
        str(batch_size),
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_images_per_prompt",
        str(num_images_per_prompt),
        "--auto_cast",
        auto_cast,
        "--auto_cast_type",
        auto_cast_type,
        output_dir,
    ]
    return compile_command


def build_pixart_command(
    hf_model_id, batch_size, sequence_length, height, width, num_images_per_prompt, torch_dtype, output_dir
):
    if None in [batch_size, sequence_length, height, width, torch_dtype]:
        raise ValueError(
            "You must provide --batch_size, --sequence_length, --height, --width and --torch_dtype for compiling pixart models."
        )
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
        "--height",
        str(height),
        "--width",
        str(width),
        "--num_images_per_prompt",
        str(num_images_per_prompt),
        "--torch_dtype",
        torch_dtype,
        output_dir,
    ]
    return compile_command


def compile_and_cache_model(
    hf_model_id: str,
    batch_size: int,
    sequence_length: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_images_per_prompt: int | None = None,
    num_cores: int | None = None,
    task: str | None = None,
    auto_cast: str | None = None,
    auto_cast_type: str | None = None,
    torch_dtype: str | None = None,
):
    start = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        if task is None:
            task = infer_task_from_model_path(hf_model_id)
        # Compile model with Optimum for specific configurations
        if task == "text-generation":
            compile_command = build_decoder_command(
                hf_model_id, batch_size, sequence_length, num_cores, auto_cast_type, temp_dir
            )
        elif "stable-diffusion" in task:
            compile_command = build_stable_diffusion_command(
                hf_model_id,
                batch_size,
                height,
                width,
                num_images_per_prompt,
                auto_cast,
                auto_cast_type,
                temp_dir,
            )
        elif "pixart" in task:
            compile_command = build_pixart_command(
                hf_model_id,
                batch_size,
                sequence_length,
                height,
                width,
                num_images_per_prompt,
                torch_dtype,
                temp_dir,
            )
        else:
            compile_command = build_encoder_command(
                hf_model_id, task, batch_size, sequence_length, auto_cast, auto_cast_type, temp_dir
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
    logger.info(f"Compiled and cached model {hf_model_id} w{time.time() - start:.2f} seconds")


def infer_task_from_model_path(model_id: str):
    try:
        # Decoder: task=="text-generation"
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        model_type = config.model_type.replace("_", "-")
        model_tasks = TasksManager.get_supported_tasks_for_model_type(
            model_type, exporter="neuron", library_name="transformers"
        )
        if "text-generation" in model_tasks:
            task = "text-generation"
            return task
    except Exception:
        pass

    # TODO: Remove when https://github.com/huggingface/optimum/pull/1793/ is merged in Optimum
    try:
        task = TasksManager.infer_task_from_model(model_id)
    except KeyError:
        model_info = huggingface_hub.model_info(model_id)
        library_name = TasksManager.infer_library_from_model(model_id)
        if library_name == "diffusers":
            class_name = model_info.config["diffusers"].get("_class_name", None)
            task = "stable-diffusion-xl" if "StableDiffusionXL" in class_name else "stable-diffusion"
    return task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and cache a model to the Hugging Face Hub.")
    parser.add_argument("--hf_model_id", type=str, help="Hugging Face model ID to compile.")
    parser.add_argument("--task", type=str, help="Task for compilation (mandatory for encoders).")
    parser.add_argument("--batch_size", type=int, help="Batch size for compilation.")
    parser.add_argument("--sequence_length", type=int, help="Sequence length for compilation.")
    parser.add_argument("--height", type=int, help="Image height for compilation.")
    parser.add_argument("--width", type=int, help="Image width for compilation.")
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt."
    )
    parser.add_argument("--num_cores", type=int, help="Number of cores for compilation.")
    parser.add_argument(
        "--auto_cast", type=str, choices=["none", "matmul", "all"], help="Operations to cast to lower precision."
    )
    parser.add_argument("--auto_cast_type", type=str, choices=["bf16", "fp16"], help="Auto cast type for compilation.")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        help="Data type (precision) of tensors used when loading a model with PyTorch.",
    )
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for authentication if not logged in.")
    parser.add_argument("--config_file", type=str, help="Path to a json config file with model configurations.")
    args = parser.parse_args()

    # Ensure either HF token is provided or user is already logged in
    original_token = get_token()
    if args.hf_token:
        logger.info(f"Logging in to Hugging Face Hub with {args.hf_token[:10]}...")
        login(args.hf_token)
    else:
        logger.info("Trying to use existing Hugging Face Hub login or environment variable HF_TOKEN")

    # check and get neuronx-cc version
    neuronx_cc_version = get_neuronxcc_version()
    sdk_version = get_aws_neuronx_tools_version()
    logger.info(f"Compiler version: {neuronx_cc_version}")
    logger.info(f"Neuron SDK version: {sdk_version}")
    logger.info(f"Optimum Neuron version: {optimum_neuron_version.__version__}")
    logger.info(f"Compatible Optimum Neuron SDK version: {optimum_neuron_version.__sdk_version__} == {sdk_version}")

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
                    hf_model_id=model_id,
                    batch_size=model_config["batch_size"],
                    sequence_length=model_config.get("sequence_length", None),
                    height=model_config.get("height", None),
                    width=model_config.get("width", None),
                    num_images_per_prompt=model_config.get("num_images_per_prompt", 1),
                    num_cores=model_config.get("num_cores", None),
                    task=model_config.get("task", None),
                    auto_cast=model_config.get("auto_cast", None),
                    auto_cast_type=model_config.get("auto_cast_type", None),
                    torch_dtype=model_config.get("torch_dtype", None),
                )
    elif args.hf_model_id is None:
        raise ValueError("You must provide --hf_model_id to compile a model without a config file.")
    else:
        compile_and_cache_model(
            hf_model_id=args.hf_model_id,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            height=args.height,
            width=args.width,
            num_images_per_prompt=args.width,
            num_cores=args.num_cores,
            task=args.task,
            auto_cast=args.auto_cast,
            auto_cast_type=args.auto_cast_type,
        )

    # Restore hub login
    if original_token:
        login(original_token)
    else:
        logout()
