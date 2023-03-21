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
"""Defines Trainer subclasses to perform training on AWS Trainium instances."""

import os
import inspect
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, Trainer

from ..utils import logging
from .utils.argument_utils import validate_arg
from .utils.cache_utils import NeuronHash, download_cached_model_from_hub, get_neuron_cache_path, list_files_in_neuron_cache, get_neuron_cache_path

from .utils.training_utils import (
    FirstAndLastDataset,
    is_model_officially_supported,
    is_precompilation,
    patch_model,
    prepare_environment_for_neuron,
)


if TYPE_CHECKING:
    from transformers import TrainingArguments, PreTrainedModel


logger = logging.get_logger(__name__)


class AugmentTrainerForTrainiumMixin:
    def __init__(self, *args, **kwargs):
        if not isinstance(self, Trainer):
            raise TypeError(f"{self.__class__.__name__} can only be mixed with Trainer subclasses.")

        training_args = kwargs.get("args", None)
        if training_args is not None:
            if training_args.bf16:
                training_args.bf16 = False
                os.environ["XLA_USE_BF16"] = "1"

        self.validate_args(training_args)
        if is_precompilation():
            self.prepare_args_for_precompilation(training_args)

        prepare_environment_for_neuron()
        super().__init__(*args, **kwargs)

        transformers_loggers = logging.get_logger("transformers.trainer")
        logger.setLevel(transformers_loggers.level)
        logger.setLevel(logging.INFO)

        # TODO: not ideal because it will not work if multiple processes are run concurrently.
        self.neuron_cache_path = get_neuron_cache_path()
        if self.neuron_cache_path is None:
            self.neuron_cache_state = None
        else:
            self.neuron_cache_state = list_files_in_neuron_cache(self.neuron_cache_path)

        if self.model is not None:
            self.fetch_precompiled_model_for_cache_repo(self.model)

    def neuron_hash_for_model(self, model: "PreTrainedModel", mode: str) -> NeuronHash:
        dataloader_method_name = f"get_{mode}_dataloader"
        dataloader = getattr(self, dataloader_method_name)()

        input_names = inspect.signature(model.forward).parameters.keys()
        batch = next(iter(dataloader))
        inputs = (batch[input_name] for input_name in batch if input_name in input_names)
        input_shapes = [input_.shape for input_ in inputs if isinstance(input_, torch.Tensor)]

        if self.args.fp16:
            data_type = torch.float16
        elif self.args.bf16:
            data_type = torch.bfloat16
        else:
            data_type = torch.float32

        return NeuronHash(
            model,
            input_shapes, 
            data_type
        )

    def fetch_precompiled_model_for_cache_repo(self, model: "PreTrainedModel"):
        original_state = model.training
        # TODO: change link to Optimum Neuron documentation page explaining the precompilation phase.
        not_found_in_cache_msg = (
            "Could not find the precompiled model on the Hub, it is recommended to run the precompilation phase, "
            "otherwise compilation will be sequential and can take some time. "
            "For more information, check here: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/finetune_hftrainer.html?highlight=precompilation#single-worker-training"
        )
        if self.args.do_train:
            model = model.train()
            neuron_hash = self.neuron_hash_for_model(model, "train")
            found_in_cache = download_cached_model_from_hub(neuron_hash)
            if not found_in_cache:
                logger.info(not_found_in_cache_msg)
        if self.args.do_eval:
            model = model.eval()
            neuron_hash = self.neuron_hash_for_model(model, "eval")
            found_in_cache = download_cached_model_from_hub(neuron_hash)
            if not found_in_cache:
                logger.info(not_found_in_cache_msg)
        if self.args.do_predict:
            model = model.eval()
            neuron_hash = self.neuron_hash_for_model(model, "test")
            found_in_cache = download_cached_model_from_hub(neuron_hash)
            if not found_in_cache:
                logger.info(not_found_in_cache_msg)

        model = model.train(original_state)

    def upload_diff_cache_path(self, model: "PreTrainedModel"):
        neuron_hash = self.neuron_hash_for_model(model, "train")


    def prepare_args_for_precompilation(self, args: "TrainingArguments"):
        if args.num_train_epochs != 1:
            logger.info("Setting the number of epochs for precompilation to 1.")
            args.num_train_epochs = 1
        if args.max_steps is not None:
            logger.info("Disabling max_steps for precompilation.")
            args.nax_steps = None
        if args.do_eval is True:
            logger.info("Disabling evaluation during precompilation as this is not well supported yet.")
            args.do_eval = False
        if args.do_predict is True:
            logger.info("Disabling prediction during precompilation as this is not well supported yet.")
            args.do_predict = False

    def validate_args(self, args: "TrainingArguments"):
        if isinstance(self, Seq2SeqTrainer):
            validate_arg(
                args,
                "prediction_loss_only",
                "prediction_loss_only=False is not supported for now because it requires generation.",
                expected_value=True,
            )

    def _wrap_model(self, model, training=True, dataloader=None):
        logger.info(
            "Disabling DDP because it is currently not playing well with multiple workers training, for more "
            "information please refer to https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/finetune_hftrainer.html#multi-worker-training"
        )
        if not is_model_officially_supported(model):
            logger.warning(
                f"{model.__class__.__name__} is not officially supported by optimum-neuron. Training might not work as  "
                "expected."
            )
        return patch_model(model)

    def get_train_dataloader(self) -> DataLoader:
        if is_precompilation():
            return DataLoader(
                FirstAndLastDataset(
                    super().get_train_dataloader(),
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    world_size=self.args.world_size,
                ),
                batch_size=None,
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if is_precompilation():
            return DataLoader(
                FirstAndLastDataset(
                    super().get_eval_dataloader(eval_dataset=eval_dataset), world_size=self.args.world_size
                ),
                batch_size=None,
            )
        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if is_precompilation():
            return DataLoader(
                FirstAndLastDataset(
                    super().get_test_dataloader(test_dataset),
                    world_size=self.args.world_size,
                ),
                batch_size=None,
            )
        return super().get_test_dataloader(test_dataset)

    def train(self, *args, **kwargs):
        res = super().train(*args, **kwargs)
        if self.neuron_cache_path is not None:
            current_neuron_cache_state = list_files_in_neuron_cache(self.neuron_cache_path)
            diff = [path for path in current_neuron_cache_state if path not in self.neuron_cache_state]
            import pdb; pdb.set_trace()
            self.neuron_cache_state = current_neuron_cache_state

        return res


class TrainiumTrainer(AugmentTrainerForTrainiumMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqTrainiumTrainer(AugmentTrainerForTrainiumMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
