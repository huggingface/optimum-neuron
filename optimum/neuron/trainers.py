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

import inspect
import json
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Seq2SeqTrainer, Trainer, TrainerCallback, TrainerState

from ..utils import logging
from .utils.argument_utils import validate_arg
from .utils.cache_utils import (
    NEURON_COMPILE_CACHE_NAME,
    NeuronHash,
    download_cached_model_from_hub,
    get_neuron_cache_path,
    list_files_in_neuron_cache,
    path_after_folder,
    push_to_cache_on_hub,
    set_neuron_cache_path,
)
from .utils.training_utils import (
    FirstAndLastDataset,
    is_model_officially_supported,
    is_precompilation,
    patch_model,
    prepare_environment_for_neuron,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TrainerControl, TrainingArguments


logger = logging.get_logger(__name__)


@dataclass
class NeuronTrainerState(TrainerState):
    # last_train_inputs: Optional[Dict[str, Any]] = None
    # last_eval_inputs: Optional[Dict[str, Any]] = None
    last_inputs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        # if self.last_train_inputs is None:
        #     self.last_train_inputs = {}
        # if self.last_eval_inputs is None:
        #     self.last_eval_inputs = {}
        if self.last_inputs is None:
            self.last_inputs = {}

    @classmethod
    def from_trainer_state(cls, state: TrainerState) -> "NeuronTrainerState":
        neuron_trainer_state = cls(asdict(state))
        # neuron_trainer_state.last_train_inputs = getattr(state, "last_train_inputs", {})
        # neuron_trainer_state.last_eval_inputs = getattr(state, "last_eval_inputs", {})
        neuron_trainer_state.last_inputs = getattr(state, "last_inputs", {})
        return neuron_trainer_state


class NeuronCacheCallaback(TrainerCallback):
    def __init__(self):
        super().__init__()
        # Real Neuron compile cache if it exists.
        self.neuron_cache_path = get_neuron_cache_path()
        self.use_neuron_cache = self.neuron_cache_path is not None
        if not self.neuron_cache_path.exists():
            self.neuron_cache_path.mkdir(parents=True)

        # Temporary Neuron compile cache.
        self.tmp_neuron_cache, self.tmp_neuron_cache_path = self.create_temporary_neuron_cache(self.neuron_cache_path)
        # self.tmp_neuron_cache_state = list(self.tmp_neuron_cache_path.iterdir())
        self.tmp_neuron_cache_state = list_files_in_neuron_cache(self.tmp_neuron_cache_path, only_relevant_files=True)
        self.fetch_files = set()

        self.neuron_hashes: Dict[Tuple["PreTrainedModel", Tuple[Tuple[int], ...], torch.dtype], NeuronHash] = {}
        self.neuron_hash_to_files: Dict[NeuronHash, List[Path]] = defaultdict(list)

    def prepare_state(self, state: TrainerState):
        if isinstance(state, NeuronTrainerState):
            return state
        return NeuronTrainerState.from_trainer_state(state)

    def get_dir_size(self, path: Path) -> int:
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")
        proc = subprocess.Popen(["du", "-s", path.as_posix()], stdout=subprocess.PIPE)
        stdout, _ = proc.communicate()
        stdout = stdout.decode("utf-8")
        return int(stdout.split()[0])

    def _load_cache_stats(self, neuron_cache_path: Path) -> Dict[str, Dict[str, Any]]:
        cache_stats_path = neuron_cache_path / "cache_stats.json"
        if cache_stats_path.exists():
            with open(neuron_cache_path / "cache_stats.json", "r") as fp:
                cache_stats = json.load(fp)
        else:
            cache_stats = {}
        return cache_stats

    def _insert_in_cache_stats(self, cache_stats: Dict[str, Dict[str, Any]], path: Path):
        path_in_cache = path_after_folder(path, NEURON_COMPILE_CACHE_NAME)
        cache_key = path_in_cache.parts[0]
        item = cache_stats.get(cache_key, {})
        if path.parent.as_posix() in item:
            return
        item[path.parent.as_posix()] = {"used_time": 1, "size": self.get_dir_size(path.parent)}
        cache_stats[cache_key] = item

    def _update_cache_stats(self, neuron_cache_path: Path):
        cache_stats = self._load_cache_stats(neuron_cache_path)
        for path in list_files_in_neuron_cache(neuron_cache_path):
            # path_in_cache = path_after_folder(path, NEURON_COMPILE_CACHE_NAME)
            self._insert_in_cache_stats(cache_stats, path)
        with open(neuron_cache_path / "cache_stats.json", "w") as fp:
            json.dump(cache_stats, fp)

    def create_temporary_neuron_cache(self, neuron_cache_path: Optional[Path]) -> Tuple[TemporaryDirectory, Path]:
        tmp_neuron_cache = TemporaryDirectory()
        tmp_neuron_cache_path = Path(tmp_neuron_cache.name)
        if neuron_cache_path is not None:
            neuron_cache_files = list_files_in_neuron_cache(neuron_cache_path)
        else:
            neuron_cache_files = []

        set_neuron_cache_path(tmp_neuron_cache_path)
        tmp_neuron_cache_path = tmp_neuron_cache_path / NEURON_COMPILE_CACHE_NAME
        tmp_neuron_cache_path.mkdir()

        cache_stats_exists = False
        if neuron_cache_path is not None:
            cache_stats = self._load_cache_stats(neuron_cache_path)
        else:
            cache_stats = {}

        for cache_file in neuron_cache_files:
            if cache_file.name == "cache_stats.json":
                continue
            path_in_neuron_cache = path_after_folder(cache_file, NEURON_COMPILE_CACHE_NAME)
            tmp_cache_file = tmp_neuron_cache_path / path_in_neuron_cache
            tmp_cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache_file.symlink_to(cache_file)

            self._insert_in_cache_stats(cache_stats, cache_file)

        if not cache_stats_exists:
            with open(tmp_neuron_cache_path / "cache_stats.json", "w") as fp:
                json.dump(cache_stats, fp)

        return tmp_neuron_cache, tmp_neuron_cache_path

    def neuron_hash_for_model(
        self,
        args: "TrainingArguments",
        model: "PreTrainedModel",
        inputs: Dict[str, Any],
        try_to_fetch_cached_model: bool = False,
    ) -> NeuronHash:
        input_names = inspect.signature(model.forward).parameters.keys()
        input_shapes = tuple(tuple(value.shape) for (input_name, value) in inputs.items() if input_name in input_names)

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
            self.neuron_hashes[key] = neuron_hash
            if try_to_fetch_cached_model:
                self.try_to_fetch_cached_model(neuron_hash)
        return neuron_hash

    def full_path_to_path_in_cache(self, path: Path):
        return path_after_folder(path, NEURON_COMPILE_CACHE_NAME)

    def try_to_fetch_cached_model(self, neuron_hash: NeuronHash) -> bool:
        files_before_fetching = list_files_in_neuron_cache(self.tmp_neuron_cache_path, only_relevant_files=True)
        cache_path = neuron_hash.cache_path

        def path_in_repo_to_path_in_target_directory(path):
            # The last part of cache_path is the overall hash.
            return Path(neuron_hash.neuron_compiler_version_dir_name) / path_after_folder(path, cache_path.name)

        found_in_cache = download_cached_model_from_hub(
            neuron_hash,
            target_directory=self.tmp_neuron_cache_path,
            path_in_repo_to_path_in_target_directory=path_in_repo_to_path_in_target_directory,
        )
        if found_in_cache and self.use_neuron_cache:
            files_after_fetching = list_files_in_neuron_cache(self.tmp_neuron_cache_path, only_relevant_files=True)
            diff = [f for f in files_after_fetching if f not in files_before_fetching]
            # The fetched files should not be synchronized with the Hub.
            self.tmp_neuron_cache_state += diff
            for path in diff:
                path_in_cache = self.full_path_to_path_in_cache(path)
                path_in_original_cache = self.neuron_cache_path / path_in_cache
                path_in_original_cache.parent.mkdir(parents=True, exist_ok=True)
                if path_in_original_cache.exists():
                    continue
                shutil.copy(path, path_in_original_cache)

        return found_in_cache

    def synchronize_temporary_neuron_cache_state(self) -> List[Path]:
        current_files_in_neuron_cache = list_files_in_neuron_cache(
            self.tmp_neuron_cache_path, only_relevant_files=True
        )
        diff = [p for p in current_files_in_neuron_cache if p not in self.tmp_neuron_cache_state]
        self.tmp_neuron_cache_state = current_files_in_neuron_cache
        return diff

    def synchronize_temporary_neuron_cache(self):
        for neuron_hash, files in self.neuron_hash_to_files.items():

            def local_path_to_path_in_repo(path):
                return path_after_folder(path, f"USER_neuroncc-{neuron_hash.neuron_compiler_version}")

            for path in files:
                push_to_cache_on_hub(neuron_hash, path, local_path_to_path_in_repo=local_path_to_path_in_repo)
                if self.use_neuron_cache:
                    path_in_cache = self.full_path_to_path_in_cache(path)
                    target_file = self.neuron_cache_path / path_in_cache
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, self.neuron_cache_path / path_in_cache)

        if self.use_neuron_cache:
            self._update_cache_stats(self.neuron_cache_path)

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        model = kwargs["model"]
        state = self.prepare_state(state)
        neuron_hash = self.neuron_hash_for_model(args, model, state.last_inputs, try_to_fetch_cached_model=True)
        diff = self.synchronize_temporary_neuron_cache_state()
        self.neuron_hash_to_files[neuron_hash].extend(diff)

    def on_save(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        """
        Event called after a checkpoint save.
        """
        self.synchronize_temporary_neuron_cache()

    def on_step_middle(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        model = kwargs["model"]
        self.neuron_hash_for_model(args, model, state.last_inputs, try_to_fetch_cached_model=True)


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

        # self.add_callback(NeuronCacheCallaback())

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

    def trigger_on_step_middle_for_neuron_cache_callback(self, model: "PreTrainedModel"):
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, NeuronCacheCallaback):
                # kwargs might not have everything expected (like metrics) but all we need is here.
                kwargs = {
                    "model": model,
                    "tokenizer": self.tokenizer,
                    "optimizer": self.optimizer,
                    "lr_scheduler": self.lr_scheduler,
                    "train_dataloader": self.callback_handler.train_dataloader,
                    "eval_dataloader": self.callback_handler.eval_dataloader,
                }
                callback.on_step_middle(self.args, self.state, self.control, **kwargs)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        self.state.last_inputs = inputs
        self.trigger_on_step_middle_for_neuron_cache_callback(model)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.state.last_inputs = inputs
        self.trigger_on_step_middle_for_neuron_cache_callback(model)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


class TrainiumTrainer(AugmentTrainerForTrainiumMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqTrainiumTrainer(AugmentTrainerForTrainiumMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
