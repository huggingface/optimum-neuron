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
"""Tools that downloads 🤗 Transformers training script examples and prepares them for AWS Tranium instances."""

import ast
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from enum import Enum, auto
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple, Union

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
    "audio-classification",
    "speech-recognition",
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


def remove_import(pattern: re.Pattern, file_content: str) -> Tuple[str, str, int]:
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


def wrap_with_lazy_load_for_parallelism(file_content: str) -> str:
    model_loading_pattern = r"\w+ = AutoModel[\w.]+"
    shift = 0
    for m in re.finditer(model_loading_pattern, file_content):
        position = m.end(0) + shift
        opened = 1
        if file_content[position] != "(":
            raise ValueError(f"Did not find an opening parenthesis, match: {m}")
        while opened > 0:
            position += 1
            if file_content[position] == ")":
                opened -= 1
            elif file_content[position] == "(":
                opened += 1

        start = m.start(0) + shift
        model_loading_content = file_content[start : position + 1]
        initial_length = len(model_loading_content)
        model_loading_content = model_loading_content.replace("\n", "\n    ")
        number_of_spaces = 0
        for i in range(start - 1, 0, -1):
            if file_content[i] == "\n":
                break
            elif file_content[i] == "\t":
                number_of_spaces += 4
            else:
                number_of_spaces += 1
        # Adding one tab to indent from the lazy_load_for_parallelism context manager.
        number_of_spaces += 4
        model_loading_content = " " * number_of_spaces + model_loading_content
        new_content = (
            "with lazy_load_for_parallelism(tensor_parallel_size=training_args.tensor_parallel_size, "
            f"pipeline_parallel_size=training_args.pipeline_parallel_size):\n{model_loading_content}\n"
        )
        file_content = file_content[:start] + new_content + file_content[position + 1 :]
        shift += len(new_content) - initial_length

    return file_content


def trim(multi_lines_string: str) -> str:
    if not multi_lines_string:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = multi_lines_string.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return "\n".join(trimmed)


class InsertPosition(Enum):
    BEFORE = auto()
    BETWEEN = auto()
    AFTER = auto()


def transform_file_content(
    file_content: str,
    predicate_func: Callable[[ast.AST], bool],
    insert_position: InsertPosition,
    code_to_insert: str,
    additional_offset: int = 0,
    num_lines_to_skip_at_match: int = 0,
) -> str:
    node = ast.parse(file_content)
    lines = file_content.split("\n")
    code_to_insert = trim(code_to_insert)
    for n in ast.walk(node):
        if predicate_func(n):
            start, end = n.lineno, n.end_lineno
            offset = n.col_offset + additional_offset
            if end is None:
                end = len(lines)
            start = max(0, start - num_lines_to_skip_at_match)
            end = min(len(lines), end + num_lines_to_skip_at_match)
            code_lines = code_to_insert.split("\n")
            # We add a carriage return at the end of the inserted code.
            code_lines = [" " * offset + line for line in code_lines]
            if insert_position is InsertPosition.BEFORE:
                # We use `start - 1` because lineno is 1 indexed.
                lines = lines[: start - 1] + code_lines + lines[start - 1 :]
            elif insert_position is InsertPosition.BETWEEN:
                lines = lines[: start - 1] + code_lines + lines[end:]
            elif insert_position is InsertPosition.AFTER:
                lines = lines[:end] + code_lines + lines[end:]
            break
    return "\n".join(lines)


def prepare_speech_script(file_content: str, seq2seq_or_ctc: str):
    assert seq2seq_or_ctc in ["seq2seq", "ctc"]
    max_label_length_data_argument = """
    max_label_length: int = field(
        default=128,
        metadata={"help": "Truncate transcriptions that are longer `max_label_length` tokens."},
    )
    """

    file_content = transform_file_content(
        file_content,
        lambda n: isinstance(n, ast.ClassDef) and n.name == "DataTrainingArguments",
        InsertPosition.AFTER,
        max_label_length_data_argument,
        # Because the predicate is based on the class definition, so we need to add a tabulation to have the proper offset.
        additional_offset=4,
    )

    xla_compatible_data_collator_for_seq2seq = """
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int
        forward_attention_mask: bool
        input_padding: Union[bool, str] = "max_length"
        target_padding: Union[bool, str] = "max_length"
        max_target_length: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need
            # different padding methods
            model_input_name = self.processor.model_input_names[0]

            # dataloader returns a list of features which we convert to a dict
            input_features = {model_input_name: [feature[model_input_name] for feature in features]}
            label_features = {"input_ids": [feature["labels"] for feature in features]}

            batch = self.processor.feature_extractor.pad(
                input_features,
                padding=self.input_padding,
                return_tensors="pt",
            )

            if self.forward_attention_mask:
                batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

            labels_batch = self.processor.tokenizer.pad(
                label_features,
                max_length=self.max_target_length,
                padding=self.target_padding,
                return_tensors="pt",
            )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch
    """
    xla_compatible_data_collator_for_ctc = """
    class DataCollatorCTCWithPadding:
        processor: AutoProcessor
        input_padding: Union[bool, str] = "max_length"
        target_padding: Union[bool, str] = "max_length"
        max_target_length: Optional[int] = None

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need
            # different padding methods
            input_features = [{"input_values": feature["input_values"]} for feature in features]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.input_padding,
                return_tensors="pt",
            )

            labels_batch = self.processor.pad(
                labels=label_features,
                max_length=self.max_target_length,
                padding=self.target_padding,
                return_tensors="pt",
            )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            if "attention_mask" in batch:
                batch["attention_mask"] = batch["attention_mask"].to(torch.long)

            return batch
    """
    file_content = transform_file_content(
        file_content,
        lambda n: isinstance(n, ast.ClassDef) and n.name == "DataCollatorSpeechSeq2SeqWithPadding",
        InsertPosition.BETWEEN,
        (
            xla_compatible_data_collator_for_seq2seq
            if seq2seq_or_ctc == "seq2seq"
            else xla_compatible_data_collator_for_ctc
        ),
    )

    import_partial_from_functools = "from functools import partial"
    file_content = transform_file_content(
        file_content,
        lambda n: isinstance(n, ast.ImportFrom) and n.module == "dataclasses",
        InsertPosition.AFTER,
        import_partial_from_functools,
    )

    filter_longer_than_max_label_length = r"""
    # filter training data with labels longer than max_label_length
    def is_labels_in_length_range(labels):
        return 0 < len(labels) < data_args.max_label_length

    vectorized_datasets = vectorized_datasets.filter(
        function=is_labels_in_length_range,
        input_columns=["labels"],
        num_proc=num_workers,
        desc="filtering dataset for labels length",
    )
    """
    file_content = transform_file_content(
        file_content,
        lambda n: isinstance(n, ast.FunctionDef) and n.name == "is_audio_in_length_range",
        InsertPosition.AFTER,
        filter_longer_than_max_label_length,
        # We skip the first 7 lines after match (which correspond to another filter) before inserting the new code.
        num_lines_to_skip_at_match=7,
    )

    data_collator_with_padding_and_max_length_for_seq2seq = """
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
          processor=processor,
          decoder_start_token_id=model.config.decoder_start_token_id,
          forward_attention_mask=forward_attention_mask,
          input_padding="longest",
          target_padding="max_length",
          max_target_length=data_args.max_label_length,
    )
    """
    data_collator_with_padding_and_max_length_for_ctc = """
    data_collator = DataCollatorCTCWithPadding(
          processor=processor,
          input_padding="longest",
          target_padding="max_length",
          max_target_length=data_args.max_label_length,
    )
    """
    file_content = transform_file_content(
        file_content,
        lambda n: isinstance(n, ast.Assign)
        and "data_collator" in [t.id for t in n.targets if isinstance(t, ast.Name)],
        InsertPosition.BETWEEN,
        (
            data_collator_with_padding_and_max_length_for_seq2seq
            if seq2seq_or_ctc == "seq2seq"
            else data_collator_with_padding_and_max_length_for_ctc
        ),
    )

    return file_content


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
                code = f"\n{code}\nfrom optimum.neuron.distributed import lazy_load_for_parallelism\n"
                processed_content = insert_code_at_position(code, processed_content, import_end_index)
                with open(training_argument_file_path, "w") as fp:
                    fp.write(processed_content)

                with open(training_argument_file_path, "r") as fp:
                    file_content = fp.read()
                processed_content = wrap_with_lazy_load_for_parallelism(file_content)
                with open(training_argument_file_path, "w") as fp:
                    fp.write(processed_content)

                if file_path.name == "run_speech_recognition_seq2seq.py":
                    with open(training_argument_file_path, "r") as fp:
                        file_content = fp.read()
                    processed_content = prepare_speech_script(file_content)
                    with open(training_argument_file_path, "w") as fp:
                        fp.write(processed_content)

                if file_path.name == "run_speech_recognition_ctc.py":
                    # TODO
                    pass

            elif file_path.name == "requirements.txt":
                with open(file_path, "r") as fp:
                    file_content = fp.read()
                processed_content = re.sub(TORCH_REQUIREMENT_PATTERN, "", file_content)
                if file_path.parent.name == "image-classification":
                    processed_content += "\nscikit-learn"
                with open(file_path, "w") as fp:
                    fp.write(processed_content)

    # Linting and styling.
    subprocess.run(["black", f"{args.dest}"])
    subprocess.run(["ruff", f"{args.dest}", "--fix"])


if __name__ == "__main__":
    main()
