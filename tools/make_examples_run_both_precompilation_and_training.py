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
"""Make the 🤗 Transformers training script examples run both precompilation and training."""

import shutil
import subprocess
from pathlib import Path
from argparse import ArgumentParser


COPYRIGHT_CONTENT = """
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
"""

IMPORT_CONTENT = """
import logging
import os
import sys

from {example_name} import main
"""


LOGGER_CONTENT = """
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
"""


MAIN_CONTENT = """
if __name__ == "__main__":
    if os.environ.get("DISABLE_PRECOMPILATION", "false") == "false" and os.environ.get("IS_PRECOMPILATION", "false") == "false":
        logger.info("Starting the precompilation phase, this may take a while...")
        os.environ["IS_PRECOMPILATION"] = "true"
        from libneuronxla.neuron_parallel_compile import main as neuron_parallel_compile_main
        original_argv = list(sys.argv)
        # TODO: handle interpreter = "torchrun"
        sys.argv = ["python"] + sys.argv
        neuron_parallel_compile_main()
        sys.argv = original_argv
        os.environ["IS_PRECOMPILATION"] = "false"
        logger.info("Precompilation done!")
    main()
"""


def transform_example(example_path: Path):
    new_example_path = example_path.parent / f"_{example_path.name}"
    shutil.move(example_path, new_example_path)
    
    with open(example_path, "w") as fp:
        transformed_example_content = "\n".join([
                COPYRIGHT_CONTENT,
                IMPORT_CONTENT.format(example_name=new_example_path.stem),
                LOGGER_CONTENT,
                MAIN_CONTENT,
        ])
        fp.write(transformed_example_content)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Tool to transform the 🤗 Transformers example training scripts to make them run both "
            "precompilation and actual training."
        )
    )
    parser.add_argument("--examples_dir", required=True,type=Path, help="The directory in which the examples will be saved.")
    parser.add_argument(
        "--examples",
        default="all",
        action="store",
        type=str,
        nargs="+",
        help=(
            "The names of the examples to transform. By default all the examples in the provided example directory "
            "will be transformed."
        )
    )
    return parser.parse_args()



def main():
    args = parse_args()
    examples_to_transform = [example for example in args.examples_dir.iterdir() if example.is_dir()]
    if args.examples != "all":
        examples_to_transform = [example for example in examples_to_transform if example.name not in args.examples]

    for example in examples_to_transform:
        for file in example.iterdir():
            if file.name.startswith("run_") and file.suffix == ".py":
                transform_example(file)
                print(f"Processed {file}")

    # Linting and styling.
    subprocess.run(["black", f"{args.examples_dir}"])
    subprocess.run(["ruff", f"{args.examples_dir}", "--fix"])

if __name__ == "__main__":
    main()
