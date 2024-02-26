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
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from huggingface_hub.utils import HfHubHTTPError
from packaging import version
from transformers import TrainerCallback, TrainerState

from ..utils import logging
from .distributed.utils import TENSOR_PARALLEL_SHARDS_DIR_NAME
from .utils import is_torch_xla_available
from .utils.cache_utils import (
    NeuronHash,
    create_or_append_to_neuron_parallel_compile_report,
    download_cached_model_from_hub,
    get_hf_hub_cache_repos,
    get_neuron_cache_path,
    get_neuron_compiler_version_dir_name,
    get_neuron_parallel_compile_report,
    has_write_access_to_repo,
    list_files_in_neuron_cache,
    path_after_folder,
    push_to_cache_on_hub,
    remove_entries_in_neuron_parallel_compile_report,
    set_neuron_cache_path,
)
from .utils.training_utils import is_precompilation
from .version import __version__


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TrainerControl, TrainingArguments

    from .training_args import NeuronTrainingArguments


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
        tmp_neuron_cache: Optional[Path] = None,
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

        cache_repo_ids = get_hf_hub_cache_repos()
        if cache_repo_ids:
            self.cache_repo_id = cache_repo_ids[0]
            has_write_access = has_write_access_to_repo(self.cache_repo_id)
            if self.push and not has_write_access:
                logger.warning(
                    f"Pushing to the remote cache repo {self.cache_repo_id} is disabled because you do not have write "
                    "access to it."
                )
                self.push = False
        else:
            self.cache_repo_id = None

        # Real Neuron compile cache if it exists.
        if original_neuron_cache_path is None:
            self.neuron_cache_path = get_neuron_cache_path()
        else:
            self.neuron_cache_path = original_neuron_cache_path
        self.use_neuron_cache = self.neuron_cache_path is not None
        self.neuron_cache_path.mkdir(parents=True, exist_ok=True)

        # Temporary Neuron compile cache.
        if is_precompilation():
            # When doing precompilation, the graph will be compiled after than the script is done.
            # By setting `self.tmp_neuron_cache` to `self.neuron_cache_path`, `neuron_parallel_compile` will extract
            # the very same graphs than the one created during real training, while not doing any synchronization
            # during training since the compiled files will not be there yet.
            self.tmp_neuron_cache_path = self.neuron_cache_path
        elif tmp_neuron_cache is None:
            # To keep an instance of the TemporaryDirectory as long as the callback lives.
            self._tmp_neuron_cache = self.create_temporary_neuron_cache(self.neuron_cache_path)
            self.tmp_neuron_cache_path = Path(self._tmp_neuron_cache.name)
        else:
            self.tmp_neuron_cache_path = tmp_neuron_cache

        self.tmp_neuron_cache_state = list_files_in_neuron_cache(self.tmp_neuron_cache_path, only_relevant_files=True)
        self.fetch_files = set()

        # Keys are of format:
        # (model, input_shapes, data_type, tensor_parallel_size)
        self.neuron_hashes: Dict[
            Tuple["PreTrainedModel", Tuple[Tuple[str, Tuple[int]], ...], torch.dtype, int], NeuronHash
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
    def _insert_in_cache_stats(cls, cache_stats: Dict[str, Dict[str, Any]], full_path: Path, path_in_cache: Path):
        cache_key = path_in_cache.parts[0]
        item = cache_stats.get(cache_key, {})
        if full_path.parent.as_posix() in item:
            return
        item[full_path.parent.as_posix()] = {"used_time": 1, "size": cls.get_dir_size(full_path.parent)}
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

        # Setting the Neuron compilation cache to be the temporary Neuron compilation cache.
        set_neuron_cache_path(tmp_neuron_cache_path)

        cache_stats_exists = False
        if neuron_cache_path is not None:
            cache_stats = cls._load_cache_stats(neuron_cache_path)
        else:
            cache_stats = {}

        for cache_file in neuron_cache_files:
            if cache_file.name == "cache_stats.json":
                continue
            try:
                path_in_neuron_cache = path_after_folder(
                    cache_file,
                    get_neuron_compiler_version_dir_name(),
                    include_folder=True,
                    fail_when_folder_not_found=True,
                )
            except Exception:
                # Here only when the folder `get_neuron_compiler_version_dir_name()` was not in the path of
                # `cache_file`. In this case, no symlink is created because it is interpreted as not being a
                # compilation file.
                continue
            tmp_cache_file = tmp_neuron_cache_path / path_in_neuron_cache
            tmp_cache_file.parent.mkdir(parents=True, exist_ok=True)
            # TODO: investigate why it is needed. Minor issue.
            if not tmp_cache_file.exists():
                tmp_cache_file.symlink_to(cache_file)

            cls._insert_in_cache_stats(cache_stats, cache_file, path_in_neuron_cache)

        if not cache_stats_exists:
            with open(tmp_neuron_cache_path / "cache_stats.json", "w") as fp:
                json.dump(cache_stats, fp)

        return tmp_neuron_cache

    def neuron_hash_for_model(
        self,
        args: "NeuronTrainingArguments",
        model: "PreTrainedModel",
        inputs: Dict[str, Any],
        try_to_fetch_cached_model: bool = False,
    ) -> NeuronHash:
        input_names = inspect.signature(model.forward).parameters.keys()
        input_shapes = tuple(
            (input_name, tuple(input_.shape)) for input_name, input_ in inputs.items() if input_name in input_names
        )

        # For backward compatibility, to not break the cache for users for now.
        if version.parse(__version__) <= version.parse("0.0.14"):
            use_bf16 = args.bf16
        else:
            use_bf16 = (
                args.bf16
                or os.environ.get("XLA_USE_BF16", "0") == "1"
                or os.environ.get("XLA_DOWNCAST_BF16", "0") == "1"
            )
        if args.fp16:
            data_type = torch.float16
        elif use_bf16:
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

        found_in_cache = download_cached_model_from_hub(
            neuron_hash,
            target_directory=self.tmp_neuron_cache_path,
            path_in_repo_to_path_in_target_directory="default",
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
            for path in files:
                push_to_cache_on_hub(
                    neuron_hash, path, cache_repo_id=self.cache_repo_id, local_path_to_path_in_repo="default"
                )
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

        if self.push or (xm.get_local_ordinal() == 0 and is_precompilation()):
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
        if xm.get_local_ordinal() == 0 and is_precompilation() and self.tmp_neuron_cache_path is not None:
            create_or_append_to_neuron_parallel_compile_report(self.tmp_neuron_cache_path, self.neuron_hash_to_files)
            for neuron_hash in self.neuron_hash_to_files:
                self.neuron_hash_to_files[neuron_hash] = []
        if self.push:
            self.synchronize_temporary_neuron_cache()
        if self.wait_for_everyone_on_push:
            xm.rendezvous("wait for everyone after pushing")

    def on_train_begin(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        """
        Event called at the beginning of training.
        """
        if is_precompilation() or self.neuron_cache_path is None:
            return
        if self.push:
            neuron_parallel_compile_report = get_neuron_parallel_compile_report(
                self.neuron_cache_path, as_neuron_hash=True
            )
            entries_to_remove = []
            for entry in neuron_parallel_compile_report:
                neuron_hash = entry["neuron_hash"]
                path = entry["directory"]
                filenames = list_files_in_neuron_cache(path, only_relevant_files=True)
                success = True
                for path in filenames:
                    try:
                        push_to_cache_on_hub(
                            neuron_hash,
                            path,
                            cache_repo_id=self.cache_repo_id,
                            local_path_to_path_in_repo="default",
                            fail_when_could_not_push=True,
                        )
                    except HfHubHTTPError:
                        # It means that we could not push, so we do not remove this entry from the report.
                        success = False
                if success:
                    entries_to_remove.append(entry)

            # Removing the entries that were uploaded.
            remove_entries_in_neuron_parallel_compile_report(self.neuron_cache_path, entries_to_remove)
        if self.wait_for_everyone_on_push:
            xm.rendezvous("wait for everyone after pushing")

    def on_train_end(self, args: "TrainingArguments", state: TrainerState, control: "TrainerControl", **kwargs):
        """
        Event called at the end of training.
        """
        self.on_save(args, state, control, **kwargs)
        if is_precompilation():
            if xm.get_local_ordinal() == 0:
                output_dir = Path(args.output_dir)
                for file_or_dir in output_dir.glob("**/*"):
                    if file_or_dir.is_file():
                        continue
                    if (
                        file_or_dir.name.startswith("checkpoint-")
                        or file_or_dir.name == TENSOR_PARALLEL_SHARDS_DIR_NAME
                    ):
                        logger.info(
                            f"Removing {file_or_dir} since the weights were produced by `neuron_parallel_compile`, "
                            "thus cannot be used."
                        )
                        shutil.rmtree(file_or_dir, ignore_errors=True)
            xm.rendezvous("wait for everyone after end of training cleanup during precompilation")

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
