# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# ==============================================================================

import hashlib
import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

from .compiler import ParallelKernel
from .module import PretrainedModel
from .ops import init_neuron


# Mainly used to expose top level APIs to the model object for serialization
class NeuronModelBase(PretrainedModel):
    def __init__(self, cpu_model):
        super().__init__()
        self.cpu_model = cpu_model

    def load_state_dict_dir(self, pretrained_model_path):
        self.cpu_model.load_state_dict_dir(pretrained_model_path)

    # top level api
    def load(self, directory):
        """Set the name of the serialization directory

        Weights will actually be loaded only when to_neuron is called.
        """
        assert self.serialization_enabled(), "serialization is not enabled for this model"
        self._compiled_artifacts_directory = directory

    # top level api
    def compile(self):
        kernels = self._get_all_kernels()
        neff_bytes_futures = {}
        parallel_degree = len(kernels)
        with ProcessPoolExecutor(parallel_degree) as executor:
            for kernel in kernels:
                neff_bytes_futures[hash_hlo(kernel.hlo_module)] = executor.submit(
                    kernel.compile, kernel.num_exec_repetition
                )
            for kernel in kernels:
                kernel.neff_bytes = neff_bytes_futures[hash_hlo(kernel.hlo_module)].result()

    # top level api
    def setup(self):
        for nbs in self.nbs_objs:
            nbs.setup()

    def load_weights(self):
        """Custom method to load model weights

        Must be implemented by the child class.
        """
        raise NotImplementedError

    def to_neuron(self):
        init_neuron()
        self.load_weights()
        if hasattr(self, "_compiled_artifacts_directory"):
            if not os.path.isdir(self._compiled_artifacts_directory):
                raise FileNotFoundError(f"Did not find directory: {self._compiled_artifacts_directory}.")
            for nbs_obj in self.nbs_objs:
                nbs_obj.set_neff_bytes(self._compiled_artifacts_directory)
        else:
            self.compile()
        self.setup()

    def save(self, directory):
        if os.path.isfile(directory):
            raise FileExistsError(f"Artifacts should be saved to a directory. Found existing file: {directory}")
        os.makedirs(directory, exist_ok=True)
        for i, nbs_obj in enumerate(self.nbs_objs):
            nbs_obj.save_compiler_artifacts(directory)

    def _get_all_kernels(self):
        all_kernels = []
        for nbs in self.nbs_objs:
            for kernel in nbs.get_all_kernels():
                all_kernels.append(kernel)
        return all_kernels

    # To enable serialization, have the model call this
    # function to register all nbs_obj of your model.
    # The nbs_obj must follow 2 rules:
    #   1. The nbs_obj must inherit from NeuronBaseSerializer.
    #   2. Since NeuronBaseSerializer is abstract, a nbs_obj.get_all_kernels()
    #      method should be implemented by the child class, which returns a
    #      list of all kernels which have NEFFs for that serialized object.
    def register_for_serialization(self, nbs_obj):
        assert issubclass(type(nbs_obj), NeuronBaseSerializer), "The nbs_obj must inherit from NeuronBaseSerializer."
        temp = getattr(self, "nbs_objs", [])
        nbs_obj.compiler_artifacts_path = None
        temp.append(nbs_obj)
        self.nbs_objs = temp

    def serialization_enabled(self):
        return getattr(self, "nbs_objs", None) is not None

    def profile(self, profile_dir, ntff_count_limit):
        kernels = self._get_all_kernels()

        for kernel in kernels:
            if isinstance(kernel, ParallelKernel):
                kernel.profile(profile_dir, ntff_count_limit)


# Base class for all "Serializable Objects"
class NeuronBaseSerializer(ABC):
    def save_compiler_artifacts(self, path):
        for kernel in self.get_all_kernels():
            hlo_hash = hash_hlo(kernel.hlo_module)
            with open(os.path.join(path, hlo_hash), "wb") as f:
                assert kernel.neff_bytes is not None, "cannot save a model which has not been successfully compiled"
                f.write(kernel.neff_bytes)

    def set_neff_bytes(self, directory):
        for kernel in self.get_all_kernels():
            hlo_hash = hash_hlo(kernel.hlo_module)
            try:
                with open(os.path.join(directory, hlo_hash), "rb") as f:
                    kernel.neff_bytes = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(
                    (
                        "Could not find a matching NEFF for your HLO in this directory. "
                        "Ensure that the model you are trying to load is the same type and "
                        'has the same parameters as the one you saved or call "save" on '
                        "this model to reserialize it."
                    )
                )

    @abstractmethod
    def get_all_kernels(self):
        raise NotImplementedError(
            f"Class {type(self)} deriving from NeuronBaseSerializer must implement get_all_kernels"
        )


def hash_hlo(hlo_module):
    hash_gen = hashlib.sha256()
    message = hlo_module.SerializeToString()
    hash_gen.update(message)
    hash = str(hash_gen.hexdigest())[:20]
    return hash + ".neff"
