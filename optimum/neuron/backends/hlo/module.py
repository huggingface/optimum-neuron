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
import json
import os
import warnings
from typing import List, Optional, Tuple

import torch
from safetensors import safe_open
from torch.nn.parameter import UninitializedParameter
from transformers import AutoConfig
from transformers.utils import hub


# Disable lazy module warning since torch-neuronx version is pinned
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.lazy")


_SAFETENSORS_MODEL_INDEX_FILENAME_JSON = "model.safetensors.index.json"
_SAFETENSORS_MODEL_FILENAME = "model.safetensors"
_PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON = "pytorch_model.bin.index.json"
_PYTORCH_MODEL_BIN_FILENAME = "pytorch_model.bin"
_KEY_TO_FILENAME_JSON = "key_to_filename.json"


class LowMemoryModule(torch.nn.Module):
    def materialize(self):
        with torch.no_grad():
            for param in self.parameters():
                if not hasattr(param, "_file_path"):
                    continue
                if param._file_path.endswith(".empty_json"):
                    with open(param._file_path) as fp:
                        empty_json = json.load(fp)
                    torch.manual_seed(0)
                    input_param = empty_json.get("init_std", 1.0) * torch.randn(empty_json["shape"])
                    dtype = getattr(torch, empty_json["torch_dtype"])
                    input_param = input_param.to(dtype)
                elif param._file_path.endswith(".safetensors"):
                    with safe_open(param._file_path, framework="pt") as f:
                        if param._global_key in f.keys():
                            input_param = f.get_tensor(param._global_key)
                        else:
                            raise FileNotFoundError(
                                f"Could not find a weight for {param._global_key} in {param._file_path}"
                            )
                else:
                    input_param = torch.load(param._file_path)
                if torch.nn.parameter.is_lazy(param):
                    param.materialize(input_param.shape)
                param.copy_(input_param)

    def nullify(self):
        def _nullify(module):
            for name, param in module.named_parameters():
                if "." not in name and hasattr(module, name):
                    blank = UninitializedParameter()
                    # Note: Allow the parameter to be reloaded
                    if hasattr(param, "_file_path"):
                        blank._file_path = param._file_path
                        blank._global_key = param._global_key
                    setattr(module, name, blank)
            for child in module.modules():
                if child is module:
                    continue
                if child is not None:
                    _nullify(child)

        _nullify(self)

    def load_state_dict_low_memory(self, state_dict):
        def load(module, prefix=""):
            module._load_from_state_dict_low_memory(state_dict, prefix)
            for name, child in module.named_modules():
                if child is module:
                    continue
                if child is not None:
                    load(child, prefix + name + ".")

        load(self)

    def _load_from_state_dict_low_memory(self, state_dict, prefix):
        local_state = {k: v for k, v in self.named_parameters() if v is not None}
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict.pop(key)
                with torch.no_grad():
                    if torch.nn.parameter.is_lazy(param):
                        param.materialize(input_param.shape)
                    param.copy_(input_param)

    def _load_state(self, state_dict_dir, key_to_filename, prefix=""):
        local_state = {k: v for k, v in self.named_parameters() if v is not None}
        complete = True
        for key, param in local_state.items():
            key = prefix + key
            if key in key_to_filename:
                path = os.path.join(state_dict_dir, key_to_filename[key])
                param._file_path = path
                param._global_key = key
            else:
                complete = False
        return complete

    def _load_ties(self):
        for ties in self.get_tied_parameters():
            # Find if any tie has a weight
            src = None
            for tie in ties:
                if hasattr(tie, "_file_path"):
                    src = tie
                    break

            if src is None:
                continue

            # Copy weight for remaining empty tied weights
            for dst in ties:
                if not hasattr(dst, "_file_path"):
                    dst._file_path = src._file_path
                    dst._global_key = src._global_key

    def _load_from_state_dict_dir(self, state_dict_dir, key_to_filename, prefix=""):
        state_dict_dir = os.path.realpath(state_dict_dir)

        # Load global weights
        complete = self._load_state(state_dict_dir, key_to_filename, prefix)

        # Check for ties and base model prefixes if initial weight load was incomplete
        if not complete:
            # Load base model weights
            base = self.get_base_model()
            if base:
                base._load_state(state_dict_dir, key_to_filename, prefix)

            # Load tied weights
            self._load_ties()

    def load_pytorch_model_bin(self, state_dict_dir):
        """
        Eagerly load the pytorch model binary artifact.
        """
        state_dict_path = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_FILENAME)
        state_dict = torch.load(state_dict_path)
        self.load_state_dict_low_memory(state_dict)

    def load_pytorch_model_bin_sharded(self, state_dict_dir):
        """
        Eagerly load the the pytorch model binary shards.
        """
        index = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON)
        with open(index, "r") as f:
            key_to_filename = json.load(f)["weight_map"]
        shard_filenames = set(key_to_filename.values())
        for shard_filename in shard_filenames:
            path = os.path.join(state_dict_dir, shard_filename)
            state_dict = torch.load(path)
            self.load_state_dict_low_memory(state_dict)

    def load_safetensors(self, state_dict_dir):
        """
        Lazily load the safetensors by associating each weight with the filename.
        """
        filename = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_FILENAME)
        with safe_open(filename, framework="pt") as f:
            keys = f.keys()
        key_to_filename = dict(zip(keys, [_SAFETENSORS_MODEL_FILENAME] * len(keys)))
        self._load_from_state_dict_dir(state_dict_dir, key_to_filename)

    def load_safetensors_sharded(self, state_dict_dir):
        """
        Lazily load the safetensors by associating each weight with a shard filename.
        """
        index = os.path.join(state_dict_dir, _SAFETENSORS_MODEL_INDEX_FILENAME_JSON)
        with open(index, "r") as f:
            key_to_filename = json.load(f)["weight_map"]
        self._load_from_state_dict_dir(state_dict_dir, key_to_filename)

    def load_split_checkpoint(self, state_dict_dir):
        """
        Lazily load the manually split checkpoint by associating each weight with the weight filename.
        """
        weight_directory = os.path.join(state_dict_dir, _PYTORCH_MODEL_BIN_FILENAME)
        with open(os.path.join(weight_directory, _KEY_TO_FILENAME_JSON)) as f:
            key_to_filename = json.load(f)
        self._load_from_state_dict_dir(weight_directory, key_to_filename)

    def get_tied_parameters(self) -> List[Tuple[torch.nn.Parameter, ...]]:
        """
        Get parameter path pairs whose weights are identical.

        Note that this is only necessary for safetensors models because at
        serialization time, `transformers` does not save duplicate/identical
        weights to the "model.safetensors" file. Instead they save only one
        weight and then check which parameters are tied according to the
        parameter structure at load time. Since we do not load the original
        parameter structure, we need an alternative method of determining
        "ties".

        Tie handling reference:
        https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/modeling_utils.py#L3884-L3925

        Tie parameter resolution reference:
        https://github.com/huggingface/accelerate/blob/v0.26.0/src/accelerate/utils/modeling.py#L530-L585
        """
        return []

    def get_base_model(self) -> Optional["LowMemoryModule"]:
        """
        Get the base pretrained transformer model.

        This information is necessary to load some checkpoint variants.

        The `base_model_prefix` was introduced very early into the
        `transformers` repository history:
        https://github.com/huggingface/transformers/commit/4d47f4985dfb09237b6e11b5eafb0b1935f8c634

        The base model prefix allows the pretrained transformer submodule
        weights to be used for the full generative model by reusing the
        embedding weight for the language model head. In most models, the
        language model head is the only difference between the underlying
        pretrained transformer and the full generative model.

        In practice, this means that some checkpoints exclude the base model
        prefix string from all serialized weight paths. Instead
        the checkpoint path names begin from a submodule within the complete
        module heirarchy. When a model is known to support a base model prefix,
        we must check if any "missing" weights would otherwise be found in the
        checkpoint if the base model prefix is removed from the expected full
        module path.

        For example, the following checkpoint paths would be interchangable with
        a base model prefix of "transformer":
        - transformer.layer[0].attention.q_proj.weight
        -             layer[0].attention.q_proj.weight

        Prefix handling reference:
        https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/modeling_utils.py#L3853-L3882
        """
        return None


class LowMemoryModuleList(torch.nn.ModuleList, LowMemoryModule): ...


class LowMemoryLazyLinear(torch.nn.LazyLinear, LowMemoryModule): ...


class LowMemoryLayerNorm(torch.nn.LayerNorm, LowMemoryModule):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super(LowMemoryLayerNorm, self).__init__(0)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = UninitializedParameter()
        self.bias = UninitializedParameter()


class LowMemoryEmbedding(torch.nn.Embedding, LowMemoryModule):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Optional[torch.Tensor] = None,
        device=None,
        dtype=None,
    ) -> None:
        super(LowMemoryEmbedding, self).__init__(0, 0)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx < 0:
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight = UninitializedParameter()


def maybe_download_weights(path_or_repo_id, safe_serialization=True, **kwargs):
    """
    Get the local path to weights or download them from the huggingface hub.

    This function makes two assumptions on behalf of the user. First it
    will prioritize the local directory and second it will prioritize
    safetensors checkpoints if they exist.

    This function does not handle remote code downloading since the assumption
    is that the model code must be implemented in transformers-neuronx.

    Args:
        path_or_repo_id: The model repository name or the local checkpoint
            path (i.e. `bigscience/bloom-560m` or `/home/ubuntu/example`)

    Kwargs:
        safe_serialization: Allow/Prohibit safetensor checkpoints.
        kwargs: Passed to `cached_file` in order to allow downloads with
            specific authorization/branches/etc.

    Returns
        directory: The local directory containing the checkpoint.
    """

    # If we have a local checkpoint, ignore remote downloading
    if os.path.isdir(path_or_repo_id):
        return path_or_repo_id

    checkpoints = [
        _SAFETENSORS_MODEL_INDEX_FILENAME_JSON,
        _SAFETENSORS_MODEL_FILENAME,
        _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON,
        _PYTORCH_MODEL_BIN_FILENAME,
    ]

    if not safe_serialization:
        checkpoints = [
            _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON,
            _PYTORCH_MODEL_BIN_FILENAME,
        ]

    # Search for checkpoints in order of fastest loads
    filename = None
    checkpoint = None
    for checkpoint in checkpoints:
        # Ignore errors since only one format may exist
        filename = hub.cached_file(
            path_or_repo_id,
            checkpoint,
            _raise_exceptions_for_missing_entries=False,
            **kwargs,
        )
        if filename:
            break

    if not filename:
        # Note: Error type matches transformers `from_pretrained` convention
        raise EnvironmentError(
            f"Could not find a checkpoint for {path_or_repo_id} in a "
            f"supported format (safetensors model or pytorch binary model)"
        )

    # Download shards if we originally found an index
    if checkpoint in [
        _SAFETENSORS_MODEL_INDEX_FILENAME_JSON,
        _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON,
    ]:
        # Do not ignore errors since we should have a sharded checkpoint
        hub.get_checkpoint_shard_files(path_or_repo_id, filename, **kwargs)

    # Return parent directory of the downloaded files
    return os.path.dirname(filename)


class PretrainedModel(LowMemoryModule):
    @classmethod
    def from_pretrained(cls, pretrained_model_path, neuron_config):
        if not os.path.isdir(pretrained_model_path):
            raise ValueError(f"{pretrained_model_path} directory does not exist.")
        config = AutoConfig.from_pretrained(pretrained_model_path)
        model = cls(config, neuron_config)
        pretrained_model_path = maybe_download_weights(pretrained_model_path)
        model.load_state_dict_dir(pretrained_model_path)
        return model

    def load_state_dict_dir(self, pretrained_model_path):
        # Standard checkpoint filenames
        state_dict_path = os.path.join(pretrained_model_path, _PYTORCH_MODEL_BIN_FILENAME)
        state_dict_safetensor_path = os.path.join(pretrained_model_path, _SAFETENSORS_MODEL_FILENAME)
        safetensors_index_path = os.path.join(pretrained_model_path, _SAFETENSORS_MODEL_INDEX_FILENAME_JSON)
        pytorch_model_bin_index_path = os.path.join(pretrained_model_path, _PYTORCH_MODEL_BIN_INDEX_FILENAME_JSON)

        # Loading is done in priority of fastest -> slowest (in case multiple variants exist)

        # Non-sharded safetensors checkpoint
        if os.path.isfile(state_dict_safetensor_path):
            self.load_safetensors(pretrained_model_path)
        # Sharded safetensors checkpoint
        elif os.path.exists(safetensors_index_path):
            self.load_safetensors_sharded(pretrained_model_path)
        # Manually split `save_pretrained_split` checkpoint
        elif os.path.isdir(state_dict_path):
            self.load_split_checkpoint(pretrained_model_path)
        # Non-sharded pytorch_model.bin checkpoint
        elif os.path.isfile(state_dict_path):
            self.load_pytorch_model_bin(pretrained_model_path)
        # Sharded pytorch model bin
        elif os.path.isfile(pytorch_model_bin_index_path):
            self.load_pytorch_model_bin_sharded(pretrained_model_path)
        else:
            raise FileNotFoundError(f"Can not find model.safetensors or pytorch_model.bin in {pretrained_model_path}")
