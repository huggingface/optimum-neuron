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
"""Defines Trainer subclasses to perform training on AWS Trainium 1 instances."""

import logging
import os
from typing import Any, Optional

from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, Trainer

from .utils import FirstAndLastDataset, patch_model, prepare_environment_for_neuron


logger = logging.getLogger(__name__)


class AugmentTrainerForTrainiumMixin:
    def __init__(self, *args, **kwargs):
        if not isinstance(self, Trainer):
            raise TypeError(f"{self.__class__.__name__} can only be mixed with Trainer subclasses.")
        prepare_environment_for_neuron()
        super().__init__(*args, **kwargs)
        self.validate_args()

    def validate_arg(self, arg_name: str, expected_value: Any, error_msg: str):
        disable_strict_mode = os.environ.get("DISABLE_STRICT_MODE", False)
        arg = getattr(self.args, arg_name, expected_value)
        if arg != expected_value:
            if disable_strict_mode:
                logger.warning(error_msg)
            else:
                raise ValueError(error_msg)

    def validate_args(self):
        self.validate_arg(
            "pad_to_max_length",
            True,
            "pad_to_max_length=False can lead to very poor performance by trigger a lot of recompilation",
        )
        if isinstance(self, Seq2SeqTrainer):
            self.validate_arg(
                "prediction_loss_only",
                True,
                "prediction_loss_only=False is not supported for now because it requires generation.",
            )
        # TODO: do we need to validate block_size (run_clm)?
        # TODO: do we need to validate val_max_target_length (run_translation)?

    def _wrap_model(self, model, training=True, dataloader=None):
        logger.info(
            "Disabling DDP because it is currently not playing well with multiple workers training, for more "
            "information please refer to https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/finetune_hftrainer.html#multi-worker-training"
        )
        return patch_model(model)

    def get_train_dataloader(self) -> DataLoader:
        if os.environ.get("IS_PRECOMPILATION", False):
            return DataLoader(
                FirstAndLastDataset(
                    super().get_train_dataloader(), gradient_accumulation_steps=self.args.gradient_accumulation_steps
                ),
                batch_size=None,
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if os.environ.get("IS_PRECOMPILATION", False):
            return DataLoader(
                FirstAndLastDataset(super().get_eval_dataloader(eval_dataset=eval_dataset)), batch_size=None
            )
        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if os.environ.get("IS_PRECOMPILATION", False):
            return DataLoader(FirstAndLastDataset(super().get_test_dataloader(test_dataset)), batch_size=None)
        return super().get_test_dataloader(test_dataset)


class TrainiumTrainer(AugmentTrainerForTrainiumMixin, Trainer):
    pass


class Seq2SeqTrainiumTrainer(AugmentTrainerForTrainiumMixin, Seq2SeqTrainer):
    pass
