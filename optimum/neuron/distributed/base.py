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
"""Base classes related to `neuronx-distributed` to perform parallelism."""

import contextlib
from abc import ABC, abstractclassmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class SavedModelInTemporaryDirectory:
    def __init__(self, model: "PreTrainedModel"):
        self.tmpdir = TemporaryDirectory()
        self.model = model

    def __enter__(self):
        self.model.save_pretrained(self.tmpdir.name)
        return self.tmpdir.name

    def __exit__(self, *exc):
        self.tmpdir.cleanup()


class ParallelModel(ABC):
    @classmethod
    @contextlib.contextmanager
    def saved_model_in_temporary_directory(cls, model: "PreTrainedModel"):
        tmpdir = TemporaryDirectory()
        model.save_pretrained(tmpdir.name)
        try:
            yield Path(tmpdir.name) / "pytorch_model.bin"
        finally:
            tmpdir.cleanup()

    @abstractclassmethod
    def parallelize(self, model: "PreTrainedModel", tp_size: int) -> "PreTrainedModel":
        pass

    def deparallelize(self, model: "PreTrainedModel") -> "PreTrainedModel":
        raise NotImplementedError
