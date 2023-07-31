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
"""Defines custom Trainer callbacks specific to AWS Neuron instances."""

import inspect
import json
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from transformers import TrainerCallback, TrainerState

from ..utils import logging
from .utils import is_torch_xla_available
from .utils.cache_utils import (
    NEURON_COMPILE_CACHE_NAME,
    NeuronHash,
    download_cached_model_from_hub,
    follows_new_cache_naming_convention,
    get_neuron_cache_path,
    list_files_in_neuron_cache,
    path_after_folder,
    push_to_cache_on_hub,
    set_neuron_cache_path,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TrainerControl, TrainingArguments


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

logger = logging.get_logger(__name__)


@dataclass
class NeuronTrainerState(TrainerState):
    last_inputs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.last_inputs is None:
            self.last_inputs = {}

    @classmethod
    def from_trainer_state(cls, state: TrainerState) -> "NeuronTrainerState":
        neuron_trainer_state = cls(asdict(state))
        neuron_trainer_state.last_inputs = getattr(state, "last_inputs", {})
        return neuron_trainer_state


class NeuronCacheCallback(TrainerCallback):
    def __init__(
        self,
        tmp_neuron_cache: Optional[TemporaryDirectory] = None,
        original_neuron_cache_path: Optional[Path] = None,
        fetch: bool = True,
        push: bool = True,
        wait_for_everyone_on_fetch: bool = True,
        wait_for_everyone_on_push: bool = True,
    ):
        super().__init__()
        self.fetch = fetch
        self.push = push
        self.wait_for_everyone_on_fetch = is_torch_xla_available() and wait_for_everyone_on_fetch
        self.wait_for_everyone_on_push = is_torch_xla_available() and wait_for_everyone_on_push

        # Real Neuron compile cache if it exists.
        if original_neuron_cache_path is None:
            self.neuron_cache_path = get_neuron_cache_path()
        else:
            self.neuron_cache_path = original_neuron_cache_path
        self.use_neuron_cache = self.neuron_cache_path is not None
        self.neuron_cache_path.mkdir(parents=True, exist_ok=True)

        # Temporary Neuron compile cache.
        if tmp_neuron_cache is None:
            self.tmp_neuron_cache = self.create_temporary_neuron_cache(self.neuron_cache_path)
        else:
            self.tmp_neuron_cache = tmp_neuron_cache
        self.tmp_neuron_cache_path = Path(self.tmp_neuron_cache.name) / NEURON_COMPILE_CACHE_NAME
        self.tmp_neuron_cache_state = list_files_in_neuron_cache(self.tmp_neuron_cache_path, only_relevant_files=True)
        self.fetch_files = set()

        self.neuron_hashes: Dict[
            Tuple["PreTrainedModel", Tuple[Tuple[str, Tuple[int]], ...], torch.dtype], NeuronHash
        ] = {}
        self.neuron_hash_to_files: Dict[NeuronHash, List[Path]] = defaultdict(list)

    def prepare_state(self, state: TrainerState):
        if isinstance(state, NeuronTrainerState):
            return state
        return NeuronTrainerState.from_trainer_state(state)

    @staticmethod
    def get_dir_size(path: Path) -> int:
        if not path.is_dir():
            raise ValueError(f"{path} is not a directory.")
        proc = subprocess.Popen(["du", "-s", path.as_posix()], stdout=subprocess.PIPE)
        stdout, _ = proc.communicate()
        stdout = stdout.decode("utf-8")
        return int(stdout.split()[0])

    @classmethod
    def _load_cache_stats(cls, neuron_cache_path: Path) -> Dict[str, Dict[str, Any]]:
        cache_stats_path = neuron_cache_path / "cache_stats.json"
        if cache_stats_path.exists():
            with open(neuron_cache_path / "cache_stats.json", "r") as fp:
                cache_stats = json.load(fp)
        else:
            cache_stats = {}
        return cache_stats

    @classmethod
    def _insert_in_cache_stats(cls, cache_stats: Dict[str, Dict[str, Any]], path: Path, cache_path: Path):
        path_in_cache = path_after_folder(path, cache_path.name)
        cache_key = path_in_cache.parts[0]
        item = cache_stats.get(cache_key, {})
        if path.parent.as_posix() in item:
            return
        item[path.parent.as_posix()] = {"used_time": 1, "size": cls.get_dir_size(path.parent)}
        cache_stats[cache_key] = item

    @classmethod
    def _update_cache_stats(cls, neuron_cache_path: Path):
        cache_stats = cls._load_cache_stats(neuron_cache_path)
        for path in list_files_in_neuron_cache(neuron_cache_path):
            cls._insert_in_cache_stats(cache_stats, path, neuron_cache_path)
        with open(neuron_cache_path / "cache_stats.json", "w") as fp:
            json.dump(cache_stats, fp)

    @classmethod
    def create_temporary_neuron_cache(cls, neuron_cache_path: Optional[Path]) -> TemporaryDirectory:
        tmp_neuron_cache = TemporaryDirectory()
        tmp_neuron_cache_path = Path(tmp_neuron_cache.name)
        if neuron_cache_path is not None:
            neuron_cache_files = list_files_in_neuron_cache(neuron_cache_path)
        else:
            neuron_cache_files = []

        if follows_new_cache_naming_convention():
            tmp_neuron_cache_path = tmp_neuron_cache_path / NEURON_COMPILE_CACHE_NAME
            set_neuron_cache_path(tmp_neuron_cache_path)
        else:
            set_neuron_cache_path(tmp_neuron_cache_path)
            tmp_neuron_cache_path = tmp_neuron_cache_path / NEURON_COMPILE_CACHE_NAME

        tmp_neuron_cache_path.mkdir()

        cache_stats_exists = False
        if neuron_cache_path is not None:
            cache_stats = cls._load_cache_stats(neuron_cache_path)
        else:
            cache_stats = {}

        for cache_file in neuron_cache_files:
            if cache_file.name == "cache_stats.json":
                continue
            path_in_neuron_cache = path_after_folder(cache_file, neuron_cache_path.name)
            tmp_cache_file = tmp_neuron_cache_path / path_in_neuron_cache
            tmp_cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache_file.symlink_to(cache_file)

            cls._insert_in_cache_stats(cache_stats, cache_file, neuron_cache_path)

        if not cache_stats_exists:
            with open(tmp_neuron_cache_path / "cache_stats.json", "w") as fp:
                json.dump(cache_stats, fp)

        return tmp_neuron_cache

    def neuron_hash_for_model(
        self,
        args: "TrainingArguments",
        model: "PreTrainedModel",
        inputs: Dict[str, Any],
        try_to_fetch_cached_model: bool = False,
    ) -> NeuronHash:
        input_names = inspect.signature(model.forward).parameters.keys()
        input_shapes = tuple(
            (input_name, tuple(input_.shape)) for input_name, input_ in inputs.items() if input_name in input_names
        )

        if args.fp16:
            data_type = torch.float16
        elif args.bf16:
            data_type = torch.bfloat16
        else:
            data_type = torch.float32

        key_args = (model, input_shapes, data_type)
        key_kwargs = {"tensor_parallel_size": args.tensor_parallel_size}
        key = key_args + tuple(key_kwargs.values())
        neuron_hash = self.neuron_hashes.get(key, None)
        if neuron_hash is None:
            neuron_hash = NeuronHash(*key_args, **key_kwargs)
            self.neuron_hashes[key] = neuron_hash
            if try_to_fetch_cached_model:
                self.try_to_fetch_cached_model(neuron_hash)
        return neuron_hash

    def full_path_to_path_in_temporary_cache(self, path: Path):
        return path_after_folder(path, self.tmp_neuron_cache_path.name)

    def try_to_fetch_cached_model(self, neuron_hash: NeuronHash) -> bool:
        # TODO: needs to be called ONLY when absolutely needed.
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
        if found_in_cache:
            files_after_fetching = list_files_in_neuron_cache(self.tmp_neuron_cache_path, only_relevant_files=True)
            diff = [f for f in files_after_fetching if f not in files_before_fetching]
            # The fetched files should not be synchronized with the Hub.
            self.tmp_neuron_cache_state += diff
            if self.use_neuron_cache:
                for path in diff:
                    path_in_cache = self.full_path_to_path_in_temporary_cache(path)
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
                if follows_new_cache_naming_convention():
                    return path_after_folder(path, f"neuronxcc-{neuron_hash.neuron_compiler_version}")
                else:
                    return path_after_folder(path, f"USER_neuroncc-{neuron_hash.neuron_compiler_version}")

            for path in files:
                push_to_cache_on_hub(neuron_hash, path, local_path_to_path_in_repo=local_path_to_path_in_repo)
                if self.use_neuron_cache:
                    path_in_cache = self.full_path_to_path_in_temporary_cache(path)
                    target_file = self.neuron_cache_path / path_in_cache
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(path, self.neuron_cache_path / path_in_cache)

        if self.use_neuron_cache:
            self._update_cache_stats(self.neuron_cache_path)

        for neuron_hash in self.neuron_hash_to_files:
            self.neuron_hash_to_files[neuron_hash] = []

    def on_step_middle(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        if self.fetch:
            model = kwargs["model"]
            self.neuron_hash_for_model(args, model, state.last_inputs, try_to_fetch_cached_model=True)
        if self.wait_for_everyone_on_fetch:
            xm.rendezvous("wait for everyone after fetching")

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        if self.push:
            model = kwargs["model"]
            state = self.prepare_state(state)
            neuron_hash = self.neuron_hash_for_model(args, model, state.last_inputs, try_to_fetch_cached_model=True)
            diff = self.synchronize_temporary_neuron_cache_state()
            self.neuron_hash_to_files[neuron_hash].extend(diff)

    def on_prediction_step(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        """
        Event called after a prediction step.
        """
        self.on_step_end(args, state, control, **kwargs)

    def on_save(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        """
        Event called after a checkpoint save.
        """
        if self.push:
            self.synchronize_temporary_neuron_cache()
        if self.wait_for_everyone_on_push:
            xm.rendezvous("wait for everyone after pushing")

    def on_train_end(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        """
        Event called at the end of training.
        """
        self.on_save(args, state, control, **kwargs)

    def on_evaluate(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        """
        Event called after an evaluation phase.
        """
        self.on_save(args, state, control, **kwargs)

    def on_predict(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        self.on_save(args, state, control, **kwargs)
