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

from dataclasses import dataclass, asdict
import os
import math
from decimal import Decimal
import inspect
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Optional, Tuple, Any, Union, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, Trainer, TrainerCallback, TrainerState
from transformers.trainer_utils import has_length

from ..utils import logging
from .utils.argument_utils import validate_arg
from .utils.cache_utils import NEURON_COMPILE_CACHE_NAME, NeuronHash, compute_file_sha256_hash, download_cached_model_from_hub, get_neuron_cache_path, list_files_in_neuron_cache, get_neuron_cache_path, push_to_cache_on_hub, set_neuron_cache_path

from .utils.training_utils import (
    FirstAndLastDataset,
    is_model_officially_supported,
    is_precompilation,
    patch_model,
    prepare_environment_for_neuron,
)


if TYPE_CHECKING:
    from transformers import TrainingArguments, PreTrainedModel, TrainerControl


logger = logging.get_logger(__name__)


@dataclass
class NeuronTrainerState(TrainerState):
    last_train_inputs: Optional[Dict[str, Any]] = None
    last_eval_inputs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.last_train_inputs is None:
            self.last_train_inputs = {}
        if self.last_eval_inputs is None:
            self.last_eval_inputs = {}

    @classmethod
    def from_trainer_state(cls, state: TrainerState) -> "NeuronTrainerState":
        neuron_trainer_state = cls(asdict(state))
        neuron_trainer_state.last_train_inputs = getattr(state, "last_train_inputs", {})
        neuron_trainer_state.last_eval_inputs = getattr(state, "last_eval_inputs", {})
        return neuron_trainer_state

class NeuronCacheCallaback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # Real Neuron compile cache if it exists.
        self.neuron_cache_path = get_neuron_cache_path()
        self.use_neuron_cache = self.neuron_cache_path is not None

        # Temporary Neuron compile cache.
        self.tmp_neuron_cache, self.tmp_neuron_cache_path = self.create_temporary_neuron_cache(self.neuron_cache_path)
        self.tmp_neuron_cache_state = list(self.tmp_neuron_cache_path.iterdir())

        self.neuron_hashes : Dict[Tuple["PreTrainedModel", Tuple[Tuple[int], ...]], data_type, str] = {}

    def prepare_state(self, state: TrainerState):
        if isinstance(state, NeuronTrainerState):
            return state
        return NeuronTrainerState.from_trainer_state(state)

    def create_temporary_neuron_cache(self, neuron_cache_path: Optional[Path]) -> Tuple[TemporaryDirectory, Path]:
        tmp_neuron_cache = TemporaryDirectory()
        tmp_neuron_cache_path = Path(tmp_neuron_cache.name)
        if neuron_cache_path is not None:
            neuron_cache_files = list_files_in_neuron_cache(neuron_cache_path)
        else:
            neuron_cache_files = []
        for cache_file in neuron_cache_files:
            if not cache_file.parent.name == NEURON_COMPILE_CACHE_NAME:
                continue
            tmp_cache_file = tmp_neuron_cache_path / cache_file.name
            tmp_cache_file.symlink_to(cache_file)
        set_neuron_cache_path(tmp_neuron_cache_path)
        return tmp_neuron_cache, tmp_neuron_cache_path

    def neuron_hash_for_model(self, args: "TrainingArguments", model: "PreTrainedModel", inputs: Dict[str, Any]) -> NeuronHash:
        input_names = inspect.signature(model.forward).parameters.keys()
        input_shapes = tuple(tuple(value.shape) for (input_name, value) in inputs.values() if input_name in input_names)

        if args.fp16:
            data_type = torch.float16
        elif args.bf16:
            data_type = torch.bfloat16
        else:
            data_type = torch.float32

        key = (model, input_shapes, data_type)
        neuron_hash = self.neuron_hashes.get(key, None)
        if neuron_hash is None:
            neuron_hash = NeuronHash(*key)
        return neuron_hash

    # def fetch_precompiled_model_for_cache_repo(self, model: "PreTrainedModel"):
    #     original_state = model.training
    #     # TODO: change link to Optimum Neuron documentation page explaining the precompilation phase.
    #     not_found_in_cache_msg = (
    #         "Could not find the precompiled model on the Hub, it is recommended to run the precompilation phase, "
    #         "otherwise compilation will be sequential and can take some time. "
    #         "For more information, check here: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/tutorials/training/finetune_hftrainer.html?highlight=precompilation#single-worker-training"
    #     )
    #     if self.args.do_train:
    #         model = model.train()
    #         neuron_hash = self.neuron_hash_for_model(model, "train")
    #         found_in_cache = download_cached_model_from_hub(neuron_hash)
    #         if not found_in_cache:
    #             logger.info(not_found_in_cache_msg)
    #     if self.args.do_eval:
    #         model = model.eval()
    #         neuron_hash = self.neuron_hash_for_model(model, "eval")
    #         found_in_cache = download_cached_model_from_hub(neuron_hash)
    #         if not found_in_cache:
    #             logger.info(not_found_in_cache_msg)
    #     if self.args.do_predict:
    #         model = model.eval()
    #         neuron_hash = self.neuron_hash_for_model(model, "test")
    #         found_in_cache = download_cached_model_from_hub(neuron_hash)
    #         if not found_in_cache:
    #             logger.info(not_found_in_cache_msg)

    #     model = model.train(original_state)

    def upload_diff_in_temporary_cache(self, model: "PreTrainedModel", mode: str):
        current_files_in_neuron_cache = list(self.tmp_neuron_cache_path.iterdir())
        diff = [p for p in current_files_in_neuron_cache if p not in self.tmp_neuron_cache_state]
        neuron_hash = self.neuron_hash_for_model(model, mode)
        for path in diff:
            print(f"Uploading {path}...")
            push_to_cache_on_hub(
                neuron_hash,
                path.as_posix(),
            )
            if self.use_neuron_cache:
                shutil.copy(path, self.neuron_cache_path / path.name)
        self.tmp_neuron_cache_state = current_files_in_neuron_cache

    def num_update_steps_per_epoch(self, args: "TrainingArguments", state: "TrainerState", **kwargs) -> int:
        train_dataloader = kwargs["train_dataloader"]
        if has_length(train_dataloader):
            num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        else:
            num_update_steps_per_epoch = state.max_steps
        return num_update_steps_per_epoch


    def steps_in_epoch(self, args: "TrainingArguments", state: "TrainerState", **kwargs) -> int: 
        if not hasattr(self, "_steps_in_epoch"):
            train_dataloader = kwargs["train_dataloader"]
            if has_length(train_dataloader):
                if 0 < args.max_steps < self.num_update_steps_per_epoch(args, state, **kwargs):
                    self._steps_in_epoch = args.max_steps
                else:
                    self._steps_in_epoch = len(train_dataloader)
            else:
                self._steps_in_epoch = args.max_steps * args.gradient_accumulation_steps
        return self._steps_in_epoch

    def inv_steps_in_epoch(self, args: "TrainingArguments", state: "TrainerState", **kwargs):
        if not hasattr(self, "_inv_steps_in_epoch"):
            self._inv_steps_in_epoch = Decimal.from_float(
                1 / self.steps_in_epoch(args, state, **kwargs)
            )
        return self._inv_steps_in_epoch

    def current_step_in_epoch(self, args: "TrainingArguments", state: "TrainerState", **kwargs) -> int:
        step = state.global_step
        steps_in_epoch = self.steps_in_epoch(args, state, **kwargs)
        num_epochs_done = math.floor(state.epoch)
        if num_epochs_done > 0:
            step = step % steps_in_epoch
        return step

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        model = kwargs["model"]
        state = self.prepare_state(state)
        neuron_hash = self.neuron_hash_for_model(args, model, state.last_train_inputs)
        if neuron_hash not in self.neuron_hashes:
            print("")

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

        self.add_callback(NeuronCacheCallaback())


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

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        self.state.last_train_inputs = inputs
        return super().compute_loss(model, inputs, return_outputs=return_outputs) 

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.state.last_eval_inputs = inputs
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


class TrainiumTrainer(AugmentTrainerForTrainiumMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqTrainiumTrainer(AugmentTrainerForTrainiumMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
