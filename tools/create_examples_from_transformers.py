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
"""Tools that downloads 🤗 Transformers training script examples and prepares them for AWS Trainium instances."""

import re
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

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


IMPORT_PATTERN = r"from transformers import \(?[\w\s,_]*?([\t ]*({class_pattern}),?\n?)((?!from)[\w\s,_])*\)?"

# TRAINER_IMPORT_PATTERN = re.compile(
#     r"from transformers import \(?[\w\s,_]*?([\t ]*((Seq2Seq)?Trainer),?\n?)((?!from)[\w\s,_])*\)?"
# )

TRAINER_IMPORT_PATTERN = re.compile(IMPORT_PATTERN.format(class_pattern="(Seq2Seq)?Trainer"))
HF_ARGUMENT_PARSER_IMPORT_PATTERN = re.compile(IMPORT_PATTERN.format(class_pattern="HfArgumentParser"))
TRAINING_ARGUMENTS_IMPORT_PATTERN = re.compile(IMPORT_PATTERN.format(class_pattern="(Seq2Seq)?TrainingArguments"))
# HF_ARGUMENT_PARSER_IMPORT_PATTERN = re.compile(
#     r"from transformers import \(?[\w\s,_]*?([\t ]*(HfArgumentParser),?\n?)((?!from)[\w\s,_])*\)?"
# )
# TRAINING_ARGUMENTS_IMPORT_PATTERN = re.compile(
#     r"from transformers import \(?[\w\s,_]*?([\t ]*(TrainingArguments),?\n?)((?!from)[\w\s,_])*\)?"
# )


TORCH_REQUIREMENT_PATTERN = re.compile(r"torch[\w\s]*([<>=!]=?\s*[\d\.]+)?\n")


AWS_CODE = {
    "Trainer": "NeuronTrainer as Trainer",
    "Seq2SeqTrainer": "Seq2SeqNeuronTrainer as Seq2SeqTrainer",
    "HfArgumentParser": "NeuronHfArgumentParser as HfArgumentParser",
    "TrainingArguments": "NeuronTrainingArguments as TrainingArguments",
    "Seq2SeqTrainingArguments": "Seq2SeqNeuronTrainingArguments as Seq2SeqTrainingArguments",
}


def download_examples_from_transformers(
    example_names: list[str],
    dest_dir: str | Path,
    predicate: Callable[[Path | None, bool]] = None,
    version: str | None = None,
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


def remove_import(pattern: re.Pattern, file_content: str) -> tuple[str, str, int]:
    match_ = re.search(pattern, file_content)
    if match_ is None:
        raise ValueError(f"Could not find a match for pattern {pattern}.")
    cls_ = match_.group(2)
    new_content = file_content[: match_.start(1)] + file_content[match_.end(1) :]
    return cls_, new_content, match_.end(0) - (match_.end(1) - match_.start(1))


def remove_trainer_import(file_content: str) -> tuple[str, str, int]:
    match_ = re.search(TRAINER_IMPORT_PATTERN, file_content)
    if match_ is None:
        raise ValueError("Could not find the import of the Trainer class from transformers.")
    trainer_cls = match_.group(2)
    new_content = file_content[: match_.start(1)] + file_content[match_.end(1) :]
    return trainer_cls, new_content, match_.end(0) - (match_.end(1) - match_.start(1))


def insert_code_at_position(code: str, file_content: str, position: int) -> str:
    return file_content[:position] + code + file_content[position:]


def generate_new_import_code(*optimum_neuron_imports: str) -> str:
    if not optimum_neuron_imports:
        raise ValueError("At least one import is expected to generate new import code.")
    import_line = ["from optimum.neuron import"]
    import_line += [f"{import_}," for import_ in optimum_neuron_imports[:-1]]
    import_line.append(optimum_neuron_imports[-1])
    return " ".join(import_line)


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
            if file_path.name == "run_generation.py":
                continue
            if "run" in file_path.name and file_path.suffix == ".py":
                if file_path.name == "run_qa.py":
                    trainer_file_path = file_path.parent / "trainer_qa.py"
                elif file_path.name == "run_seq2seq_qa.py":
                    trainer_file_path = file_path.parent / "trainer_seq2seq_qa.py"
                else:
                    trainer_file_path = file_path
                hf_argument_file_path = file_path
                training_argument_file_path = file_path

                print(f"Processing {file_path}")
                with open(trainer_file_path, "r") as fp:
                    file_content = fp.read()
                trainer_cls, processed_content, import_end_index = remove_import(TRAINER_IMPORT_PATTERN, file_content)
                code = generate_new_import_code(AWS_CODE[trainer_cls])
                code = f"\n{code}\n"
                processed_content = insert_code_at_position(code, processed_content, import_end_index)
                with open(trainer_file_path, "w") as fp:
                    fp.write(processed_content)

                with open(hf_argument_file_path, "r") as fp:
                    file_content = fp.read()
                _, processed_content, import_end_index = remove_import(HF_ARGUMENT_PARSER_IMPORT_PATTERN, file_content)
                code = generate_new_import_code(AWS_CODE["HfArgumentParser"])
                code = f"\n{code}\n"
                processed_content = insert_code_at_position(code, processed_content, import_end_index)
                with open(hf_argument_file_path, "w") as fp:
                    fp.write(processed_content)

                with open(training_argument_file_path, "r") as fp:
                    file_content = fp.read()
                training_args_cls, processed_content, import_end_index = remove_import(
                    TRAINING_ARGUMENTS_IMPORT_PATTERN, file_content
                )
                code = generate_new_import_code(AWS_CODE[training_args_cls])
                code = f"\n{code}\n"
                processed_content = insert_code_at_position(code, processed_content, import_end_index)
                with open(training_argument_file_path, "w") as fp:
                    fp.write(processed_content)

            elif file_path.name == "requirements.txt":
                with open(file_path, "r") as fp:
                    file_content = fp.read()
                processed_content = re.sub(TORCH_REQUIREMENT_PATTERN, "", file_content)
                if file_path.parent.name == "image-classification":
                    processed_content += "\nscikit-learn"
                with open(file_path, "w") as fp:
                    fp.write(processed_content)

    # Linting and styling.
    subprocess.run(["ruff", f"{args.dest}", "--fix"])


if __name__ == "__main__":
    main()
