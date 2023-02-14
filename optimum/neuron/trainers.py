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

from .utils import FirstAndLastDataset, prepare_environment_for_neuron


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
        # TODO: do we need to validate block_size (run_clm)?
        # TODO: do we need to validate val_max_target_length (run_translation)?

    # def _wrap_model(self, model, training=True, dataloader=None):
    #     # Fixup to enable distributed training with XLA
    #     # TODO: investigate on that => might cause issue with gradient accumulation.
    #     # Workaround for NaNs seen with transformers version >= 4.21.0
    #     # https://github.com/aws-neuron/aws-neuron-sdk/issues/593
    #     if os.environ.get("XLA_USE_BF16") or os.environ.get("XLA_DOWNCAST_BF16"):
    #         return patch_model(model)
    #     return model

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
