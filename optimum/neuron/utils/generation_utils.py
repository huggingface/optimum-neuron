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
"""Generation utilities."""

from functools import wraps
from typing import Any, Callable, Dict

import torch
from transformers import GenerationConfig

from .import_utils import is_torch_xla_available
from .misc import args_and_kwargs_to_kwargs_only


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


def move_dict_args_to_device(kwargs: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
    """
    Takes keyword arguments which will be passed to a model's forward function
    and moves its values to `device` if
    they are of type `torch.Tensor`. If the key is a dictionary it does the same to the
    respective values.

    Args:
        kwargs: (`Dict[str, Any]`):
            The kwargs to be passed to the models forward function.
        device: (`str`, defaults to `cpu`):
            The target device to which tensors should be moved.

    Returns:
        `Dict[str, Any]`: The kwargs dict with its tensors moved to `device`.
    """

    def needs_move(src_device, tgt_device):
        return src_device != tgt_device

    for k, v in kwargs.items():
        # Handle nested dicts
        if isinstance(v, dict):
            for k_, v_ in v.items():
                if isinstance(v_, torch.Tensor):
                    if needs_move(v_.device, device):
                        v[k_] = v_.to(device=device)

        # Handle tensor types
        elif isinstance(v, torch.Tensor):
            if needs_move(v.device, device):
                kwargs[k] = v.to(device=device)

        # Handle past_key_value tuples
        elif k == "past_key_values":
            if v is not None:
                new_past_key_values = ()
                for layer_past in v:
                    new_layer_past = ()
                    for past_state in layer_past:
                        if needs_move(past_state.device, device):
                            new_layer_past += (past_state.to(device=device),)
                        else:
                            new_layer_past += (past_state,)
                    new_past_key_values += (new_layer_past,)
                kwargs[k] = new_past_key_values

    return kwargs


def pad_input_ids_for_general_sampling(
    input_ids: torch.Tensor, num_padding_values: int, pad_token_id: int
) -> torch.Tensor:
    """
    Pads `input_ids` with `num_padding_values` padding tokens along the second dimension.

    Args:
        input_ids (`torch.Tensor`):
            Input ids to be padded.
        num_padding_values (`int`):
            Number of padding values to add.
        pad_token_id (`int`):
            Token ID of padding token.

    Returns:
        `torch.Tensor`: Padded `input_ids`.
    """
    bsz = input_ids.size(0)
    input_ids = torch.cat(
        [input_ids, torch.ones((bsz, num_padding_values), device=input_ids.device, dtype=torch.long) * pad_token_id], 1
    )
    return input_ids


def get_fwd_for_general_sampling(
    current_fwd: Callable, generation_config: GenerationConfig, main_device: str, to_device: str = "cpu"
) -> Callable:
    """
    Wraps the passed forward function and extends it such that before each forward call
    the `decoder_input_ids` are padded and all tensors are moved to `main_device` (e.g. XLA).
    Then the original forward passed is called followed by a `xm.mark_step`. Subsequently,
    an "unpadding" of the logits is performed. This way, all functions that process the logits
    can be called without making any changes.

    Args:
        current_fwd (`Callable`):
            The current forward function of the model.
        generation_config (`GenerationConfig`):
            The GenerationConfig of the model.
        main_device (`str`):
            The device on which the forward pass should be executed.
        to_device (`str`, defaults to `cpu`):
            The device on which all other processing should be executed.

    Returns:
        `Callable`: The extended forward function.
    """

    @wraps(current_fwd)
    def new_fwd(*args, **kwargs):
        # Pad input to max length
        cur_len = None
        if "decoder_input_ids" in kwargs:
            current_input_ids = kwargs["decoder_input_ids"]
            cur_len = current_input_ids.size(1)
            num_padding_values = generation_config.max_length - cur_len
            kwargs["decoder_input_ids"] = pad_input_ids_for_general_sampling(
                current_input_ids, num_padding_values, generation_config.pad_token_id
            )

        # Move inputs to device
        move_dict_args_to_device(kwargs, main_device)

        # Forward
        kwargs = args_and_kwargs_to_kwargs_only(current_fwd, args, kwargs)
        outputs = current_fwd(**kwargs)
        xm.mark_step()

        # Move to CPU
        move_dict_args_to_device(outputs, to_device)

        # Post-process output as a function of cur_len
        outputs["logits"] = outputs["logits"][:, :cur_len, ...]

        return outputs

    return new_fwd
