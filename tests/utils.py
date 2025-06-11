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
"""Various utilities used in multiple tests."""

import contextlib
import functools
import inspect
import os
import random
import string
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from datasets import Dataset, DatasetDict
from huggingface_hub import CommitOperationDelete, HfApi, create_repo, delete_repo, get_token, login, logout
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
)
from transformers.testing_utils import ENDPOINT_STAGING

from optimum.neuron import NeuronAccelerator
from optimum.neuron.distributed import lazy_load_for_parallelism
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.utils.cache_utils import (
    delete_custom_cache_repo_name_from_hf_home,
    load_custom_cache_repo_name_from_hf_home,
    set_custom_cache_repo_name_in_hf_home,
)
from optimum.neuron.utils.patching import DynamicPatch, Patcher
from optimum.neuron.utils.require_utils import requires_neuronx_distributed
from optimum.utils import logging


logger = logging.get_logger(__name__)


# Not critical, only usable on the sandboxed CI instance.
USER_STAGING = "__DUMMY_OPTIMUM_USER__"
TOKEN_STAGING = "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"

SEED = 42
OPTIMUM_INTERNAL_TESTING_CACHE_REPO = "optimum-internal-testing/optimum-neuron-cache-for-testing"
OPTIMUM_INTERNAL_TESTING_CACHE_REPO_FOR_CI = "optimum-internal-testing/optimum-neuron-cache-ci"

MODEL_NAME = "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"


def get_random_string(length) -> str:
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for _ in range(length))


def create_dummy_dataset(input_specs: Dict[str, Tuple[Tuple[int, ...], torch.dtype]], num_examples: int) -> Dataset:
    def gen():
        for _ in range(num_examples):
            yield {name: torch.rand(shape) for name, shape in input_specs.items()}

    return Dataset.from_generator(gen)


def create_dummy_text_classification_dataset(
    num_train_examples: int, num_eval_examples: int, num_test_examples: Optional[int]
) -> DatasetDict:
    if num_test_examples is None:
        num_test_examples = num_eval_examples

    def create_gen(num_examples, with_labels: bool = True):
        def gen():
            for _ in range(num_examples):
                yield {
                    "sentence": get_random_string(random.randint(64, 256)),
                    "labels": random.randint(0, 1) if with_labels else -1,
                }

        return gen

    ds = DatasetDict()
    ds["train"] = Dataset.from_generator(create_gen(num_train_examples))
    ds["eval"] = Dataset.from_generator(create_gen(num_eval_examples))
    ds["test"] = Dataset.from_generator(create_gen(num_test_examples, with_labels=False))

    return ds


def generate_input_ids(vocab_size: int, batch_size: int, sequence_length: int) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, sequence_length))


def generate_attention_mask(batch_size: int, sequence_length: int, random: bool = False) -> torch.Tensor:
    if random:
        return torch.randint(0, 2, (batch_size, sequence_length))
    return torch.ones((batch_size, sequence_length))


def create_dummy_causal_lm_dataset(
    vocab_size: int,
    num_train_examples: int,
    num_eval_examples: int,
    num_test_examples: Optional[int] = None,
    max_number_of_unique_examples: Optional[int] = None,
    sequence_length: int = 32,
    random_attention_mask: bool = False,
) -> DatasetDict:
    if num_test_examples is None:
        num_test_examples = num_eval_examples

    if max_number_of_unique_examples is None:
        max_number_of_unique_examples = max(num_train_examples, num_eval_examples, num_test_examples)

    def create_gen(num_examples):
        def gen():
            examples = []
            for _ in range(min(num_examples, max_number_of_unique_examples)):
                input_ids = generate_input_ids(vocab_size, 1, sequence_length)
                attention_mask = generate_attention_mask(1, sequence_length, random=random_attention_mask)
                examples.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": input_ids,
                    }
                )
            for i in range(num_examples):
                yield examples[i % max_number_of_unique_examples]

        return gen

    ds = DatasetDict()
    ds["train"] = Dataset.from_generator(create_gen(num_train_examples))
    ds["eval"] = Dataset.from_generator(create_gen(num_eval_examples))
    ds["test"] = Dataset.from_generator(create_gen(num_test_examples))

    return ds


def default_data_collator_for_causal_lm(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    feature_names = features[0].keys()
    return {k: torch.cat([torch.tensor(feature[k]) for feature in features], dim=0) for k in feature_names}


def static_initializer_seed(initialization_function: Callable, seed: int):
    @functools.wraps(initialization_function)
    def wrapper(*args, **kwargs):
        from transformers import set_seed

        set_seed(seed)
        return initialization_function(*args, **kwargs)

    return wrapper


class StaticSeedPatcher:
    """
    Context manager that resets the seed to a given value for every initialization function.
    This is useful because lazy initialization works but does not respect the random state of the non-lazy case.
    This allows us to test that lazy initialization works if we ignore the random seed.
    """

    def __init__(self, seed: int):
        specialized_static_initializer_seed = functools.partial(static_initializer_seed, seed=seed)
        dynamic_patch = DynamicPatch(specialized_static_initializer_seed)
        self.patcher = Patcher(
            [
                # (fully_qualified_method_name, dynamic_patch),
                ("torch.nn.Embedding.reset_parameters", dynamic_patch),
                ("torch.nn.Linear.reset_parameters", dynamic_patch),
                ("torch.Tensor.normal_", dynamic_patch),
                ("neuronx_distributed.parallel_layers.layers.ColumnParallelLinear.init_weight_cpu", dynamic_patch),
                ("neuronx_distributed.parallel_layers.layers.RowParallelLinear.init_weight_cpu", dynamic_patch),
                (
                    "neuronx_distributed.modules.qkv_linear.GQAQKVColumnParallelLinear._init_per_layer_weight",
                    dynamic_patch,
                ),
                (
                    "neuronx_distributed.modules.qkv_linear.GQAQKVColumnParallelLinear._init_per_layer_bias",
                    dynamic_patch,
                ),
            ]
        )

    def __enter__(self, *args, **kwargs):
        self.patcher.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        self.patcher.__exit__(*args, **kwargs)


def get_model(
    model_class: Type["PreTrainedModel"],
    model_name_or_path: str,
    tp_size: int = 1,
    pp_size: int = 1,
    lazy_load: bool = False,
    from_config: bool = False,
    use_static_seed_patcher: bool = False,
    add_random_noise: bool = False,
    config_overwrite: Optional[Dict[str, str]] = None,
) -> "PreTrainedModel":
    if lazy_load:
        ctx = lazy_load_for_parallelism(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    else:
        ctx = contextlib.nullcontext()
    if use_static_seed_patcher:
        seed_patcher = StaticSeedPatcher(SEED)
    else:
        seed_patcher = contextlib.nullcontext()
    with ctx:
        with seed_patcher:
            config = AutoConfig.from_pretrained(model_name_or_path)
            if config_overwrite is not None:
                for key, value in config_overwrite.items():
                    attr_type = type(getattr(config, key))
                    setattr(config, key, attr_type(value))
            if from_config:
                model = model_class(config)
            else:
                model = model_class.from_pretrained(model_name_or_path, config=config, ignore_mismatched_sizes=True)

    if getattr(model.config, "problem_type", None) is None:
        model.config.problem_type = "single_label_classification"

    if add_random_noise:
        for param in model.parameters():
            param.data.add_(torch.randn_like(param))

    return model


@requires_neuronx_distributed
def generate_dummy_labels(
    model: "PreTrainedModel",
    shape: List[int],
    vocab_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """Generates dummy labels."""
    from neuronx_distributed.pipeline import NxDPPModel

    if isinstance(model, NxDPPModel):
        model_class_name = model.original_torch_module.__class__.__name__
    else:
        model_class_name = model.__class__.__name__

    labels = {}

    batch_size = shape[0]

    if model_class_name in [
        *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
        *get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES),
        *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
        *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
        *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
    ]:
        labels["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif model_class_name in [
        *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
        *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
        "XLNetForQuestionAnswering",
    ]:
        labels["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif model_class_name in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
        if not hasattr(model.config, "problem_type") or model.config.problem_type is None:
            raise ValueError(
                "Could not retrieve the problem type for the sequence classification task, please set "
                'model.config.problem_type to one of the following values: "regression", '
                '"single_label_classification", or "multi_label_classification".'
            )

        if model.config.problem_type == "regression":
            labels_shape = (batch_size, model.config.num_labels)
            labels_dtype = torch.float32
        elif model.config.problem_type == "single_label_classification":
            labels_shape = (batch_size,)
            labels_dtype = torch.long
        elif model.config.problem_type == "multi_label_classification":
            labels_shape = (batch_size, model.config.num_labels)
            labels_dtype = torch.float32
        else:
            raise ValueError(
                'Expected model.config.problem_type to be either: "regression", "single_label_classification"'
                f', or "multi_label_classification", but "{model.config.problem_type}" was provided.'
            )
        labels["labels"] = torch.zeros(*labels_shape, dtype=labels_dtype, device=device)
    elif model_class_name in [
        *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
        *get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES),
        *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES),
        "GPT2DoubleHeadsModel",
        "PeftModelForCausalLM",
        "PeftModelForSeq2SeqLM",
    ]:
        if vocab_size is None:
            raise ValueError(
                "The vocabulary size needs to be specified to generate dummy labels for language-modeling tasks."
            )
        if seed is not None:
            orig_seed = torch.seed()
            torch.manual_seed(seed)
        if model_class_name in get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES):
            max_value = model.config.num_labels
        else:
            max_value = vocab_size
        random_labels = torch.randint(0, max_value, shape, dtype=torch.long)
        if device is not None:
            random_labels = random_labels.to(device)
        labels["labels"] = random_labels
        if seed is not None:
            torch.manual_seed(orig_seed)
    elif model_class_name in [*get_values(MODEL_FOR_CTC_MAPPING_NAMES)]:
        labels["labels"] = torch.zeros(shape, dtype=torch.float32, device=device)
    else:
        raise NotImplementedError(f"Generating the dummy input named for {model_class_name} is not supported yet.")
    return labels


def get_model_inputs(
    model: "PreTrainedModel",
    model_name_or_path: str,
    include_labels: bool = True,
    random_labels: bool = True,
    batch_size: int = 1,
    pad_to_multiple_of: Optional[int] = None,
):
    input_str = "Hello there, I'm Michael and I live in Paris!"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    inputs = tokenizer(input_str, return_tensors="pt")

    if model.config.is_encoder_decoder:
        sig = inspect.signature(model.forward)
        for input_name in inputs:
            decoder_input_name = f"decoder_{input_name}"
            if decoder_input_name in sig.parameters:
                inputs[decoder_input_name] = inputs[input_name].clone()

    if include_labels:
        if random_labels:
            labels = generate_dummy_labels(model, inputs["input_ids"].shape, vocab_size=model.config.vocab_size)
            inputs.update(**labels)
        else:
            labels = tokenizer(input_str, return_tensors="pt")["input_ids"]
            inputs["labels"] = labels

    if batch_size > 1:
        for name, tensor in inputs.items():
            repeat = [batch_size] + [1] * (tensor.dim() - 1)
            tensor = tensor.repeat(*repeat)
            inputs[name] = tensor

    if pad_to_multiple_of is not None:
        pad_token_id = getattr(model.config, "pad_token_id", 1)
        for name, tensor in inputs.items():
            if tensor.dim() == 2 and tensor.shape[1] % pad_to_multiple_of != 0:
                if "attention_mask" not in name:
                    pad_value = pad_token_id
                else:
                    pad_value = 1
                tensor = torch.nn.functional.pad(
                    tensor,
                    pad=(0, pad_to_multiple_of - tensor.shape[1] % pad_to_multiple_of),
                    value=pad_value,
                )
                inputs[name] = tensor
    return inputs


def get_tokenizer_and_tiny_llama_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, ignore_mismatched_sizes=True)
    return tokenizer, model


def create_accelerator(
    tp_size: int,
    pp_size: int,
    zero_1: bool = False,
    gradient_accumulation_steps: int = 1,
    parallelize_embeddings: bool = True,
    sequence_parallel_enabled: bool = True,
    kv_size_multiplier: Optional[int] = None,
    checkpoint_dir: Optional[Union[Path, str]] = None,
    use_xser: bool = True,
) -> NeuronAccelerator:
    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        kv_size_multiplier=kv_size_multiplier,
        parallelize_embeddings=parallelize_embeddings,
        sequence_parallel_enabled=sequence_parallel_enabled,
        pipeline_parallel_size=pp_size,
        checkpoint_dir=checkpoint_dir,
        use_xser=use_xser,
    )
    return NeuronAccelerator(
        trn_config=trn_config, zero_1=zero_1, gradient_accumulation_steps=gradient_accumulation_steps
    )


class TrainiumTestMixin:
    @classmethod
    def setUpClass(cls):
        cls._token = get_token()
        cls._cache_repo = load_custom_cache_repo_name_from_hf_home()
        cls._env = dict(os.environ)

    @classmethod
    def tearDownClass(cls):
        os.environ = cls._env
        if cls._token is not None:
            login(cls._token)
        if cls._cache_repo is not None:
            try:
                set_custom_cache_repo_name_in_hf_home(cls._cache_repo)
            except Exception:
                logger.warning(f"Could not restore the cache repo back to {cls._cache_repo}")
        else:
            delete_custom_cache_repo_name_from_hf_home()


class StagingTestMixin:
    CUSTOM_CACHE_REPO_NAME = "optimum-neuron-cache-testing"
    CUSTOM_CACHE_REPO = f"{USER_STAGING}/{CUSTOM_CACHE_REPO_NAME}"
    CUSTOM_PRIVATE_CACHE_REPO = f"{CUSTOM_CACHE_REPO}-private"
    _token = ""
    MAX_NUM_LINEARS = 20

    @classmethod
    def set_hf_hub_token(cls, token: Optional[str]) -> Optional[str]:
        orig_token = get_token()
        login(token=token)
        if token is not None:
            login(token=token)
        else:
            logout()
        cls._env = dict(os.environ, HF_ENDPOINT=ENDPOINT_STAGING)
        return orig_token

    @classmethod
    def setUpClass(cls):
        cls._staging_token = TOKEN_STAGING
        cls._token = cls.set_hf_hub_token(TOKEN_STAGING)
        cls._custom_cache_repo_name = load_custom_cache_repo_name_from_hf_home()
        delete_custom_cache_repo_name_from_hf_home()

        # Adding a seed to avoid concurrency issues between staging tests.
        cls.seed = get_random_string(5)
        cls.CUSTOM_CACHE_REPO = f"{cls.CUSTOM_CACHE_REPO}-{cls.seed}"
        cls.CUSTOM_PRIVATE_CACHE_REPO = f"{cls.CUSTOM_PRIVATE_CACHE_REPO}-{cls.seed}"

        create_repo(cls.CUSTOM_CACHE_REPO, repo_type="model", exist_ok=True)
        create_repo(cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model", exist_ok=True, private=True)

        # We store here which architectures we already used for compiling tiny models.
        cls.visited_num_linears = set()

    @classmethod
    def tearDownClass(cls):
        delete_repo(repo_id=cls.CUSTOM_CACHE_REPO, repo_type="model")
        delete_repo(repo_id=cls.CUSTOM_PRIVATE_CACHE_REPO, repo_type="model")
        if cls._token:
            cls.set_hf_hub_token(cls._token)
        if cls._custom_cache_repo_name:
            try:
                set_custom_cache_repo_name_in_hf_home(cls._custom_cache_repo_name)
            except Exception:
                logger.warning(f"Could not restore the cache repo back to {cls._custom_cache_repo_name}")
            set_custom_cache_repo_name_in_hf_home(cls._custom_cache_repo_name, check_repo=False)

    def remove_all_files_in_repo(self, repo_id: str):
        api = HfApi()
        filenames = api.list_repo_files(repo_id=repo_id)
        operations = [CommitOperationDelete(path_in_repo=filename) for filename in filenames]
        try:
            api.create_commit(
                repo_id=repo_id,
                operations=operations,
                commit_message="Cleanup the repo",
            )
        except RepositoryNotFoundError:
            pass

    def tearDown(self):
        login(TOKEN_STAGING)
        self.remove_all_files_in_repo(self.CUSTOM_CACHE_REPO)
        self.remove_all_files_in_repo(self.CUSTOM_PRIVATE_CACHE_REPO)
