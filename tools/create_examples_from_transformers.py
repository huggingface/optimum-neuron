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
"""Tools that downloads 🤗 Transformers training script examples and prepares them for AWS Tranium instances."""

import re
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Union

from git import Repo


REPO_URL = "https://github.com/huggingface/transformers.git"

SUPPORTED_EXAMPLES = [
    "text-classification",
    "token-classification",
    "question-answering",
    "multiple-choice",
    "image-classification",
    "language-modeling",
    "translation",
    "summarization",
]

UNSUPPORTED_SCRIPTS_FOR_NOW = [
    "run_plm.py",
    "run_qa_beam_search.py",
]


IMPORT_PATTERN = re.compile(
    r"from transformers import \(?[\w\s,_]*?([\t ]*((Seq2Seq)?Trainer),?\n?)((?!from)[\w\s,_])*\)?"
)

TORCH_REQUIREMENT_PATTERN = re.compile(r"torch[\w\s]*([<>=!]=?\s*[\d\.]+)?\n")

AWS_CODE = {
    "Trainer": "from optimum.neuron import TrainiumTrainer as Trainer",
    "Seq2SeqTrainer": "from optimum.neuron import Seq2SeqTrainiumTrainer as Seq2SeqTrainer",
}


def download_examples_from_transformers(
    example_names: List[str],
    dest_dir: Union[str, Path],
    predicate: Optional[Callable[[Path], bool]] = None,
    version: Optional[str] = None,
):
    if isinstance(dest_dir, str):
        dest_dir = Path(dest_dir)

    if predicate is None:
        def predicate(_):
            return True

    with TemporaryDirectory() as tmpdirname:
        repo = Repo.clone_from(REPO_URL, tmpdirname)
        if version is not None:
            pattern = rf"v{version}-(release|patch)"
            match_ = re.search(pattern, repo.git.branch("--all"))
            if match_ is None:
                raise ValueError(f"Could not find the {version} version in the Transformers repo.")
            repo.git.checkout(match_.group(0))

        path_prefix = Path(tmpdirname) / "examples" / "pytorch"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for example in example_names:
            example_dir = path_prefix / example
            for file_path in example_dir.iterdir():
                if predicate(file_path):
                    dest_example_dir = dest_dir / example
                    dest_example_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file_path, dest_example_dir / file_path.name)


def keep_only_examples_with_trainer_and_requirements_predicate(file_path: Path) -> bool:
    is_python_or_text = file_path.suffix in [".py", ".txt"]
    is_supported = file_path.name not in UNSUPPORTED_SCRIPTS_FOR_NOW
    not_a_no_trainer_script = "no_trainer" not in file_path.name
    is_requirements = file_path.name == "requirements.txt"
    return is_python_or_text and is_supported and (not_a_no_trainer_script or is_requirements)


def remove_trainer_import(file_content: str) -> tuple[str, str, int]:
    match_ = re.search(IMPORT_PATTERN, file_content)
    if match_ is None:
        raise ValueError("Could not find the import of the Trainer class from transformers.")
    trainer_cls = match_.group(2)
    new_content = file_content[: match_.start(1)] + file_content[match_.end(1) :]
    return trainer_cls, new_content, match_.end(0) - (match_.end(1) - match_.start(1))


def insert_code_at_position(code: str, file_content: str, position: int) -> str:
    return file_content[:position] + code + file_content[position:]


def parse_args():
    parser = ArgumentParser(
        description="Tool to download and prepare 🤗 Transformers example training scripts for AWS Trainium instances."
    )
    parser.add_argument(
        "--version",
        default=None,
        type=str,
        help="The version of Transformers from which the examples will be downloaded. By default the main branch is used.",
    )
    parser.add_argument(
        "--examples",
        default="all",
        action="store",
        type=str,
        nargs="+",
        help="The names of the examples to download. By default all the supported examples will be downloaded.",
    )
    parser.add_argument("dest", type=Path, help="The directory in which the examples will be saved.")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = args.examples
    if examples == "all":
        examples = SUPPORTED_EXAMPLES
    download_examples_from_transformers(
        examples, args.dest, predicate=keep_only_examples_with_trainer_and_requirements_predicate, version=args.version
    )

    for example_dir in args.dest.iterdir():
        if example_dir.is_file():
            continue
        for file_path in example_dir.iterdir():
            if "run" in file_path.name and file_path.suffix == ".py":
                print(f"Processing {file_path}")
                if file_path.name == "run_qa.py":
                    file_path = file_path.parent / "trainer_qa.py"
                elif file_path.name == "run_seq2seq_qa.py":
                    file_path = file_path.parent / "trainer_seq2seq_qa.py"
                with open(file_path, "r") as fp:
                    file_content = fp.read()
                trainer_cls, processed_content, import_end_index = remove_trainer_import(file_content)
                code = f"\n{AWS_CODE[trainer_cls]}\n"
                processed_content = insert_code_at_position(code, processed_content, import_end_index)
                with open(file_path, "w") as fp:
                    fp.write(processed_content)
            elif file_path.name == "requirements.txt":
                with open(file_path, "r") as fp:
                    file_content = fp.read()
                processed_content = re.sub(TORCH_REQUIREMENT_PATTERN, "", file_content)
                with open(file_path, "w") as fp:
                    fp.write(processed_content)

    # Linting and styling.
    subprocess.run(["black", f"{args.dest}"])
    subprocess.run(["ruff", f"{args.dest}", "--fix"])


if __name__ == "__main__":
    main()
