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
"""Script to cache diffusion models for inference."""

import argparse
import json
import logging
import subprocess
import tempfile
import time

import requests
from optimum.exporters.tasks import TasksManager

from optimum.neuron.utils.instance import SUPPORTED_INSTANCE_TYPES


# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()


def build_stable_diffusion_command(
    hf_model_id, instance_type, batch_size, height, width, num_images_per_prompt, auto_cast, auto_cast_type, output_dir
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
        "--instance_type",
        instance_type,
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
    hf_model_id,
    instance_type,
    batch_size,
    sequence_length,
    height,
    width,
    num_images_per_prompt,
    torch_dtype,
    output_dir,
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
        "--instance_type",
        instance_type,
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
    instance_type: str,
    batch_size: int,
    sequence_length: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_images_per_prompt: int | None = None,
    task: str | None = None,
    auto_cast: str | None = None,
    auto_cast_type: str | None = None,
    torch_dtype: str | None = None,
):
    start = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        if task is None:
            task = TasksManager.infer_task_from_model(hf_model_id)
        # Compile model with Optimum for specific configurations
        if "stable-diffusion" in task:
            compile_command = build_stable_diffusion_command(
                hf_model_id,
                instance_type,
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
                instance_type,
                batch_size,
                sequence_length,
                height,
                width,
                num_images_per_prompt,
                torch_dtype,
                temp_dir,
            )
        else:
            raise ValueError(f"Unsupported task {task}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and cache a model to the Hugging Face Hub.")
    parser.add_argument("--hf_model_id", type=str, help="Hugging Face model ID to compile.")
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
    parser.add_argument("--height", type=int, help="Image height for compilation.")
    parser.add_argument("--width", type=int, help="Image width for compilation.")
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1, help="Number of images to generate per prompt."
    )
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
                    hf_model_id=model_id,
                    instance_type=model_config.get("instance_type", SUPPORTED_INSTANCE_TYPES[0]),
                    batch_size=model_config["batch_size"],
                    sequence_length=model_config.get("sequence_length", None),
                    height=model_config.get("height", None),
                    width=model_config.get("width", None),
                    num_images_per_prompt=model_config.get("num_images_per_prompt", 1),
                    task=model_config.get("task", None),
                    auto_cast=model_config.get("auto_cast", None),
                    auto_cast_type=model_config.get("auto_cast_type", None),
                    torch_dtype=model_config.get("dtype", None) or model_config.get("torch_dtype", None),
                )
    elif args.hf_model_id is None:
        raise ValueError("You must provide --hf_model_id to compile a model without a config file.")
    else:
        compile_and_cache_model(
            hf_model_id=args.hf_model_id,
            instance_type=args.instance_type,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            height=args.height,
            width=args.width,
            num_images_per_prompt=args.width,
            task=args.task,
            auto_cast=args.auto_cast,
            auto_cast_type=args.auto_cast_type,
            torch_dtype=args.torch_dtype,
        )
