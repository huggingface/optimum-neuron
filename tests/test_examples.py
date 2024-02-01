# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
"""Tests every (architecture, task) supported pair on 🤗 Transformers training example scripts."""

import json
import os
import re
import sys
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union
from unittest import TestCase

from huggingface_hub import get_token
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
)
from transformers.testing_utils import slow

from optimum.neuron.distributed.parallelizers_manager import ParallelizersManager
from optimum.neuron.utils.cache_utils import load_custom_cache_repo_name_from_hf_home
from optimum.neuron.utils.import_utils import is_neuronx_distributed_available
from optimum.neuron.utils.misc import string_to_bool
from optimum.neuron.utils.runner import ExampleRunner
from optimum.neuron.utils.testing_utils import is_trainium_test

from .utils import TrainiumTestMixin


# Doing it this way to be able to use this file in tools.
path_tests = Path(__file__).parent
sys.path.insert(0, str(path_tests))

T = TypeVar("T")
TypeOrDictOfType = Union[T, Dict[str, T]]


TOKEN = get_token()
if os.environ.get("HF_TOKEN", None) is not None:
    TOKEN = os.environ.get("HF_TOKEN")

DEFAULT_CACHE_REPO = "optimum-internal-testing/optimum-neuron-cache-for-testing"
SAVED_CUSTOM_CACHE_REPO = load_custom_cache_repo_name_from_hf_home()
CUSTOM_CACHE_REPO = os.environ.get("CUSTOM_CACHE_REPO", None)
if SAVED_CUSTOM_CACHE_REPO is None and CUSTOM_CACHE_REPO is None:
    os.environ["CUSTOM_CACHE_REPO"] = DEFAULT_CACHE_REPO


class TPSupport(str, Enum):
    """
    Describes the support for Tensor Parallelism for a given model:

        - full: The model can be fully parallelized (embeddings + blocks + cross-entropy loss when it makes sense).
        - partial: The model can be parallelized but not the embeddings. Usually because the vocabulary size is not
        divisible by the tensor parallel size (2 here).
        - none: The model cannot be parallelized, either for shape mismatch as in the partial case, or because the
        tensor parallelism support for this model type has not been added.
    """

    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class Coverage(str, Enum):
    LOW = "low"
    MIDDLE = "middle"
    HIGH = "high"
    ALL = "all"


USE_VENV = string_to_bool(os.environ.get("USE_VENV", "false"))
COVERAGE = Coverage(os.environ.get("COVERAGE", "all"))
RUN_TINY = string_to_bool(os.environ.get("RUN_TINY", "false"))

MODELS_TO_TEST_MAPPING = {
    "albert": (
        "albert-base-v2",
        TPSupport.NONE,
        Coverage.LOW,
        {"num_hidden_layers": 4},
    ),
    "bart": (
        "facebook/bart-base",
        TPSupport.NONE,
        Coverage.MIDDLE,
        {"encoder_layers": 2, "decoder_layers": 2},
    ),
    "bert": (
        "bert-base-uncased",
        TPSupport.FULL,
        Coverage.HIGH,
        {"num_hidden_layers": 4},
    ),
    "camembert": (
        "camembert-base",
        TPSupport.NONE,
        Coverage.LOW,
        {"num_hidden_layers": 4},
    ),
    "distilbert": (
        "distilbert-base-uncased",
        TPSupport.NONE,
        Coverage.LOW,
        {"num_hidden_layers": 4},
    ),
    "electra": (
        "google/electra-base-discriminator",
        TPSupport.NONE,
        Coverage.LOW,
        {"num_hidden_layers": 4},
    ),
    "gpt2": (
        "gpt2",
        TPSupport.NONE,
        Coverage.MIDDLE,
        {"num_hidden_layers": 4},
    ),
    "gpt_neo": (
        "EleutherAI/gpt-neo-125M",
        TPSupport.PARTIAL,
        Coverage.HIGH,
        {"num_hidden_layers": 4, "attention_types": [[["global", "local"], 2]]},
    ),
    "marian": (
        "Helsinki-NLP/opus-mt-en-ro",
        TPSupport.NONE,
        Coverage.MIDDLE,
        {"encoder_layers": 2, "decoder_layers": 2},
    ),
    "roberta": (
        "roberta-base",
        TPSupport.PARTIAL,
        Coverage.LOW,
        {"num_hidden_layers": 4},
    ),
    "t5": (
        "t5-small",
        TPSupport.FULL,
        Coverage.HIGH,
        {"num_hidden_layers": 2},
    ),
    "vit": (
        "google/vit-base-patch16-224-in21k",
        TPSupport.NONE,
        Coverage.HIGH,
        {"num_hidden_layers": 4},
    ),
    "xlm-roberta": (
        "xlm-roberta-base",
        TPSupport.NONE,
        Coverage.LOW,
        {"num_hidden_layers": 4},
    ),
    # TODO: issue with this model for now.
    "m2m_100": (
        "facebook/m2m100_418M",
        TPSupport.NONE,
        Coverage.MIDDLE,
        {"encoder_layers": 2, "decoder_layers": 2},
    ),
    "llama": (
        "NousResearch/Llama-2-7b-hf",
        TPSupport.FULL,
        Coverage.HIGH,
        {"num_hidden_layers": 2},
    ),
    "mistral": (
        "mistralai/Mistral-7B-v0.1",
        TPSupport.FULL,
        Coverage.HIGH,
        {"num_hidden_layers": 2},
    ),
    # "wav2vec2": "facebook/wav2vec2-base",
    # Remaning: XLNet, Deberta-v2, MPNet, CLIP
}


def _get_supported_models_for_script(
    models_to_test: Dict[str, str], task_mapping: Dict[str, str], to_exclude: Optional[Set[str]] = None
) -> List[Tuple[str, str, TPSupport, Dict[str, Any]]]:
    """
    Filters models that can perform the task from models_to_test.
    """
    if to_exclude is None:
        to_exclude = set()
    supported_models = []
    for model_type, entry in models_to_test.items():
        model_name, tp_support, coverage, config_overrides = entry
        if model_type in to_exclude:
            continue
        if COVERAGE != "all" and COVERAGE != coverage:
            continue
        if CONFIG_MAPPING[model_type] in task_mapping:
            if model_type == "bart" and task_mapping is not MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
                continue
            supported_models.append((model_type, model_name, tp_support, config_overrides))
    return supported_models


_SCRIPT_TO_MODEL_MAPPING = {
    "run_clm": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING,
        MODEL_FOR_CAUSAL_LM_MAPPING,
        to_exclude={"bart", "bert", "camembert", "electra", "roberta", "xlm-roberta"},
    ),
    "run_mlm": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MASKED_LM_MAPPING),
    "run_swag": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_MULTIPLE_CHOICE_MAPPING),
    "run_qa": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, to_exclude={"gpt2", "gpt_neo", "bart", "t5"}
    ),
    "run_summarization": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, to_exclude={"marian", "m2m_100"}
    ),
    "run_translation": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    ),
    "run_glue": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING, to_exclude={"gpt2", "gpt_neo", "bart", "t5"}
    ),
    "run_ner": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, to_exclude={"gpt2", "gpt_neo"}
    ),
    "run_image_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    ),
    "run_audio_classification": _get_supported_models_for_script(
        MODELS_TO_TEST_MAPPING, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING
    ),
    "run_speech_recognition_ctc": _get_supported_models_for_script(MODELS_TO_TEST_MAPPING, MODEL_FOR_CTC_MAPPING),
}


class ExampleTestMeta(type):
    """
    Metaclass that takes care of creating the proper example tests for a given task.

    It uses example_name to figure out which models support this task, and create a run example test for each of these
    models.
    """

    def __new__(cls, name, bases, attrs, example_name=None):
        models_to_test = []
        if example_name is not None:
            models_to_test = _SCRIPT_TO_MODEL_MAPPING.get(example_name)
            if models_to_test is None:
                raise AttributeError(f"could not create class because no model was found for example {example_name}")
        for model_type, model_name_or_path, tp_support, config_overrides in models_to_test:
            # Regular training.
            attrs[f"test_{example_name}_{model_type}"] = cls._create_test(
                model_type, model_name_or_path, 1, 1, True, False, config_overrides
            )

            # Training with ZeRO-1.
            # TODO: enable this once fix from #222 is merged.
            # attrs[f"test_{example_name}_{model_type}_with_zero1"] = cls._create_test(
            #     model_type, model_name_or_path, 1, True, True, config_overrides
            # )

            tensor_parallel_size = 2 if tp_support is not TPSupport.NONE else 1

            if not is_neuronx_distributed_available():
                pp_support = False
            else:
                pp_support = ParallelizersManager.parallelizer_for_model(model_type).supports_pipeline_parallelism()
            pipeline_parallel_size = 4 if pp_support else 1

            disable_embedding_parallelization = tp_support is TPSupport.PARTIAL
            if tensor_parallel_size > 1:
                # Training with TP if supported.
                attrs[f"test_{example_name}_{model_type}_with_tp_only"] = cls._create_test(
                    model_type,
                    model_name_or_path,
                    tensor_parallel_size,
                    1,  # No pipeline parallelism in this test.
                    disable_embedding_parallelization,
                    False,
                    config_overrides,
                )
                # Training with TP and ZeRO-1 if supported.
                # TODO: enable this once fix from #222 is merged.
                # attrs[f"test_{example_name}_{model_type}_with_tp_and_zero1"] = cls._create_test(
                #     model_type,
                #     model_name_or_path,
                #     tensor_parallel_size,
                #     1, # No pipeline parallelism in this test.
                #     disable_embedding_parallelization,
                #     True,
                #     config_overrides,
                # )

            if pipeline_parallel_size > 1:
                # Training with PP if supported.
                attrs[f"test_{example_name}_{model_type}_with_pp_only"] = cls._create_test(
                    model_type,
                    model_name_or_path,
                    1,  # No tensor parallelism in this test.
                    pipeline_parallel_size,
                    disable_embedding_parallelization,
                    False,
                    config_overrides,
                )

            if tensor_parallel_size > 1 and pipeline_parallel_size > 1:
                attrs[f"test_{example_name}_{model_type}_with_tp_and_pp"] = cls._create_test(
                    model_type,
                    model_name_or_path,
                    tensor_parallel_size,
                    pipeline_parallel_size,
                    disable_embedding_parallelization,
                    False,
                    config_overrides,
                )
                # TODO: enable when working on the multi-node training PR.
                # attrs[f"test_{example_name}_{model_type}_with_tp_and_pp_and_zero1"] = cls._create_test(
                #     model_type,
                #     model_name_or_path,
                #     tensor_parallel_size,
                #     pipeline_parallel_size,
                #     disable_embedding_parallelization,
                #     True,
                #     config_overrides,
                # )

        attrs["EXAMPLE_NAME"] = example_name
        return super().__new__(cls, name, bases, attrs)

    @staticmethod
    def process_class_attribute(attribute: Union[Any, Dict[str, Any]], model_type: str) -> Any:
        if isinstance(attribute, dict):
            return attribute.get(model_type, attribute["default"])
        return attribute

    @staticmethod
    def parse_loss_from_log(log: str) -> List[float]:
        pattern = re.compile(r"{'loss': ([0-9]+\.[0-9]+),.*?}")
        losses = []
        for match_ in re.finditer(pattern, log):
            losses.append(float(match_.group(1)))
        return losses

    @staticmethod
    def check_that_loss_is_decreasing(
        losses: List[float], window_size: int, allowed_miss_rate: float = 0.1
    ) -> Tuple[bool, List[float]]:
        def moving_average(values: List[float], window_size: int):
            averages = []
            n = len(values)
            for i in range(n - window_size + 1):
                window = values[i : i + window_size]
                averages.append(sum(window) / window_size)
            return averages

        moving_average_losses = moving_average(losses, window_size)
        num_losses = len(moving_average_losses)
        num_misses = 0
        num_misses_allowed = int(num_losses * allowed_miss_rate)
        for x, y in zip(moving_average_losses[:-1], moving_average_losses[1:]):
            if x > y:
                num_misses += 1

        return num_misses <= num_misses_allowed, moving_average_losses

    @classmethod
    def _create_test(
        cls,
        model_type: str,
        model_name_or_path: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        disable_embedding_parallelization: bool,
        zero_1: bool,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Callable[["ExampleTesterBase"], None]:
        """
        Creates a test function that runs an example for a model_name.

        Returns:
            `Callable[[ExampleTesterBase], None]`: The test function that runs the example.
        """

        @slow
        @is_trainium_test
        def test(self):
            train_batch_size = ExampleTestMeta.process_class_attribute(self.TRAIN_BATCH_SIZE, model_type)
            eval_batch_size = ExampleTestMeta.process_class_attribute(self.EVAL_BATCH_SIZE, model_type)
            sequence_length = ExampleTestMeta.process_class_attribute(self.SEQUENCE_LENGTH, model_type)
            gradient_accumulation_steps = ExampleTestMeta.process_class_attribute(
                self.GRADIENT_ACCUMULATION_STEPS, model_type
            )

            runner = ExampleRunner(
                model_name_or_path,
                self.TASK_NAME,
                example_dir=self.EXAMPLE_DIR,
                use_venv=USE_VENV,
                config_overrides=config_overrides if RUN_TINY else None,
            )

            # TP = 2, NUM_CORES = 32 (DP = 16) seems to be an unsupported topology.
            num_cores = 8 if tensor_parallel_size > 1 else self.NUM_CORES

            with TemporaryDirectory() as tmpdirname:
                returncode, stdout = runner.run(
                    num_cores,
                    "bf16",
                    train_batch_size,
                    sequence_length=sequence_length,
                    do_eval=self.DO_EVAL,
                    eval_batch_size=eval_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    num_epochs=self.NUM_EPOCHS,
                    max_steps=self.MAX_STEPS,
                    max_eval_samples=self.MAX_EVAL_SAMPLES,
                    save_steps=self.SAVE_STEPS,
                    save_total_limit=1,
                    learning_rate=self.LEARNING_RATE,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    disable_embedding_parallelization=disable_embedding_parallelization,
                    zero_1=zero_1,
                    output_dir=tmpdirname,
                    do_precompilation=True,
                    print_outputs=True,
                    _disable_is_private_model_repo_check=True,
                )
                assert returncode == 0

                if self.CHECK_THAT_LOSS_IS_DECREASING:
                    losses = ExampleTestMeta.parse_loss_from_log(stdout)
                    allowed_miss_rate = 0.20
                    is_decreasing, moving_average_losses = ExampleTestMeta.check_that_loss_is_decreasing(
                        # The loss might stagnate at some point, so we only check that the first 200 losses are
                        # decreasing on average.
                        losses[200:],
                        4,
                        allowed_miss_rate=allowed_miss_rate,
                    )
                    self.assertTrue(
                        is_decreasing,
                        f"The moving average loss does not decrease as expected: {moving_average_losses} (allowed miss "
                        "rate = {allowed_miss_rate})",
                    )

                if not RUN_TINY and self.DO_EVAL:
                    with open(Path(tmpdirname) / "all_results.json") as fp:
                        results = json.load(fp)
                    eval_score_threshold = ExampleTestMeta.process_class_attribute(
                        self.EVAL_SCORE_THRESHOLD, model_type
                    )
                    if self.EVAL_SCORE_GREATER_IS_BETTER:
                        self.assertGreaterEqual(float(results[self.SCORE_NAME]), eval_score_threshold)
                    else:
                        self.assertLessEqual(float(results[self.SCORE_NAME]), eval_score_threshold)

        return test


class ExampleTesterBase(TrainiumTestMixin, TestCase):
    """
    Base example tester class.
    """

    EXAMPLE_DIR = Path(__file__).parent.parent / "examples"
    TASK_NAME: str

    NUM_EPOCHS: int = 1
    MAX_STEPS: Optional[int] = None

    LEARNING_RATE: TypeOrDictOfType[float] = 1e-4
    TRAIN_BATCH_SIZE: TypeOrDictOfType[int] = 2
    EVAL_BATCH_SIZE: TypeOrDictOfType[int] = 2
    GRADIENT_ACCUMULATION_STEPS: TypeOrDictOfType[int] = 1
    SEQUENCE_LENGTH: TypeOrDictOfType[Optional[Union[int, Tuple[int, int], List[int]]]] = None

    NUM_CORES: int = 32
    LOGGING_STEPS: int = 1
    SAVE_STEPS: int = 200

    TRAIN_LOSS_THRESHOLD: float
    CHECK_THAT_LOSS_IS_DECREASING: TypeOrDictOfType[bool] = True

    # Camembert is pretrained on French.
    DO_EVAL: TypeOrDictOfType[bool]
    MAX_EVAL_SAMPLES: Optional[int] = None
    EVAL_SCORE_THRESHOLD: TypeOrDictOfType[float]
    EVAL_SCORE_GREATER_IS_BETTER: bool
    SCORE_NAME: str


class CausalLMExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_clm"):
    TASK_NAME = "causal-lm"

    MAX_STEPS = 200

    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    SEQUENCE_LENGTH = 512

    TRAIN_LOSS_THRESHOLD = 1.5

    DO_EVAL = False
    MAX_EVAL_SAMPLES = 200


class TextClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_glue"):
    TASK_NAME = "text-classification"

    MAX_STEPS = 200

    SEQUENCE_LENGTH = 128

    TRAIN_LOSS_THRESHOLD = 0.5

    # Camembert is pretrained on French.
    DO_EVAL = False  # TODO: Evaluation is broken.
    MAX_EVAL_SAMPLES = 200
    EVAL_SCORE_THRESHOLD = {"default": 0.75, "camembert": 0.5}
    EVAL_SCORE_GREATER_IS_BETTER = True
    SCORE_NAME = "eval_accuracy"


class TokenClassificationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_ner"):
    TASK_NAME = "token-classification"

    NUM_EPOCHS = 1
    MAX_STEPS = 200

    TRAIN_BATCH_SIZE = {"default": 4, "distilbert": 6}
    EVAL_BATCH_SIZE = {"default": 4, "distilbert": 6}
    SEQUENCE_LENGTH = 384

    TRAIN_LOSS_THRESHOLD = 0.5

    # Camembert is pretrained on French.
    DO_EVAL = False  # TODO: Evaluation is broken.
    MAX_EVAL_SAMPLES = 200
    EVAL_SCORE_THRESHOLD = {"default": 0.75, "camembert": 0.5}
    EVAL_SCORE_GREATER_IS_BETTER = True
    SCORE_NAME = "eval_accuracy"


class MultipleChoiceExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_swag"):
    TASK_NAME = "multiple-choice"

    MAX_STEPS = 200

    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    SEQUENCE_LENGTH = 512

    TRAIN_LOSS_THRESHOLD = 0.5

    # Camembert is pretrained on French.
    DO_EVAL = False  # TODO: Evaluation is broken.
    MAX_EVAL_SAMPLES = 200
    EVAL_SCORE_THRESHOLD = {"default": 0.75, "camembert": 0.5, "distilbert": 0.645}
    EVAL_SCORE_GREATER_IS_BETTER = True
    SCORE_NAME = "eval_accuracy"


class QuestionAnsweringExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_qa"):
    TASK_NAME = "question-answering"

    MAX_STEPS = 200

    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    SEQUENCE_LENGTH = 384

    TRAIN_LOSS_THRESHOLD = 0.5

    DO_EVAL = False  # TODO: Evaluation is broken.
    MAX_EVAL_SAMPLES = 200
    EVAL_SCORE_THRESHOLD = {"default": 0.75, "camembert": 0.5}
    EVAL_SCORE_GREATER_IS_BETTER = True
    SCORE_NAME = "eval_f1"


class SummarizationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_summarization"):
    TASK_NAME = "summarization"

    MAX_STEPS = 200

    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    SEQUENCE_LENGTH = {"default": [1024, 200], "t5": [768, 200]}

    TRAIN_LOSS_THRESHOLD = 0.5

    DO_EVAL = False  # TODO: Evaluation is broken.
    MAX_EVAL_SAMPLES = 200
    EVAL_SCORE_THRESHOLD = 30
    SCORE_NAME = "eval_rougeLsum"


class TranslationExampleTester(ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_translation"):
    TASK_NAME = "translation"

    MAX_STEPS = 200

    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2
    SEQUENCE_LENGTH = {"default": [512, 512], "m2m_100": [128, 128]}

    DO_EVAL = False
    MAX_EVAL_SAMPLES = 200
    EVAL_SCORE_THRESHOLD = 22
    SCORE_NAME = "eval_bleu"


class ImageClassificationExampleTester(
    ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_image_classification"
):
    TASK_NAME = "image-classification"

    MAX_STEPS = 200

    TRAIN_BATCH_SIZE = 2
    EVAL_BATCH_SIZE = 2

    TRAIN_LOSS_THRESHOLD = 0.5

    DO_EVAL = False  # TODO: Evaluation is broken.
    MAX_EVAL_SAMPLES = 200
    EVAL_SCORE_THRESHOLD = 0.8
    EVAL_SCORE_GREATER_IS_BETTER = True
    SCORE_NAME = "eval_accuracy"


# class AudioClassificationExampleTester(
#     ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_audio_classification"
# ):
#     TASK_NAME = "superb"
#     DATASET_CONFIG_NAME = "ks"
#     GRADIENT_ACCUMULATION_STEPS = 16
#     EXTRA_COMMAND_LINE_ARGUMENTS = ["--max_length_seconds 1", "--attention_mask False"]
#     LEARNING_RATE = 3e-5
#
#
# class SpeechRecognitionExampleTester(
#     ExampleTesterBase, metaclass=ExampleTestMeta, example_name="run_speech_recognition_ctc"
# ):
#     TASK_NAME = "common_voice"
#     DATASET_CONFIG_NAME = "tr"
#     TRAIN_BATCH_SIZE = 1
#     GRADIENT_ACCUMULATION_STEPS = 8
#     EVAL_BATCH_SIZE = 1
#     NUM_EPOCHS = 15
#     # Here we are evaluating against the loss because it can take a long time to have wer < 1.0
#     SCORE_NAME = "eval_loss"
#     EVAL_SCORE_THRESHOLD = 4
#     EVAL_SCORE_GREATER_IS_BETTER = False
#     EXTRA_COMMAND_LINE_ARGUMENTS = [
#         "--learning_rate 3e-4",
#         "--warmup_steps 400",
#         "--mask_time_prob 0.0",
#         "--layerdrop 0.0",
#         "--freeze_feature_encoder",
#         "--text_column_name sentence",
#         "--length_column_name input_length",
#         '--chars_to_ignore , ? . ! - \\; \\: \\" “ % ‘ ” � ',
#     ]
