# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# Wrappers used at runtime to prepare inputs for decoder models.

import logging

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from ...config import NxDNeuronConfig
from ...model_wrapper import NxDModelWrapper


CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"
SPECULATION_MODEL_TAG = "speculation_model"


class NxDDecoderWrapperForCausalLM(NxDModelWrapper):
    """A decoder wrapper for decoder models used in causal language modeling.

    It prepares the inputs tensors to match the compiled model static input shapes.
    """

    def __init__(
        self, config: PretrainedConfig, neuron_config: NxDNeuronConfig, model: torch.jit.ScriptModule, tag: str
    ) -> None:
        super().__init__()
        self.config = config
        self.neuron_config = neuron_config
        self.model = model
        self.tag = tag

        if not self.neuron_config.torch_dtype:
            self.neuron_config.torch_dtype = torch.float32

        if config.pad_token_id is None:
            config.pad_token_id = 0

    def _forward_with_pad(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        # pad the inputs up to the compiled batch size in the end
        def pad_helper(tensor, pad_type="zeros"):
            VALID_PAD_TYPES = {"zeros", "ones", "repeat_first_batchline"}
            assert pad_type in VALID_PAD_TYPES, f"Found {pad_type=}, but valid pad types are {VALID_PAD_TYPES}"
            if tensor is None or tensor.shape[0] == self.neuron_config.batch_size:
                return tensor

            padded_shape = list(tensor.shape)
            padded_shape[0] = self.neuron_config.batch_size
            if pad_type == "repeat_first_batchline":
                # pad with first batch line values instead of zeros, to reduce chances of NaN
                padded_tensor = tensor[0].unsqueeze(0).repeat(padded_shape[0], 1).to(tensor.dtype)
            else:
                fill_value = 0 if pad_type == "zeros" else 1
                padded_tensor = torch.full(padded_shape, fill_value=fill_value, dtype=tensor.dtype)
            padded_tensor[: tensor.shape[0]] = tensor
            return padded_tensor

        padded_args = []
        for arg in (input_ids, attention_mask, position_ids):
            padded_args.append(pad_helper(arg, pad_type="repeat_first_batchline"))

        # need to handle seq_ids separately, when compiled batch is 4, if we pad seq_ids from [0,2,1] to [0,2,1,
        # 0]. then the kv cache of padded input could be written into the first cache line, so we need to pad as [0,
        # 2, 1, 3] instead

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list + [x for x in range(self.neuron_config.max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        padded_args.append(padded_seq_ids)

        # pad sampling params by repeating first batchline
        padded_sampling_params = pad_helper(sampling_params, pad_type="repeat_first_batchline")
        padded_args.append(padded_sampling_params)

        outputs = self._forward(*padded_args)

        # note that we don't do index select here as it should already be handled, simply sliced out padding here
        logits = outputs
        return logits[: seq_ids.shape[0]]

    def _forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        needs_reordering = False
        if self.tag == TOKEN_GENERATION_MODEL_TAG and self.neuron_config.continuous_batching:
            # if continuous batching is enabled, we need to ensure that the inputs are at the expected positions
            orig_seq_ids = seq_ids.clone()
            needs_reordering = not torch.equal(seq_ids, torch.arange(seq_ids.shape[0]))
            if needs_reordering:
                sorting_index = torch.argsort(seq_ids)
                seq_ids = torch.index_select(seq_ids, 0, sorting_index)
                input_ids = torch.index_select(input_ids, 0, sorting_index)
                attention_mask = torch.index_select(attention_mask, 0, sorting_index)
                position_ids = torch.index_select(position_ids, 0, sorting_index)
                sampling_params = torch.index_select(sampling_params, 0, sorting_index)
        outputs = self.model(input_ids, attention_mask, position_ids, seq_ids, sampling_params)
        if needs_reordering:
            # if we reordered the inputs, we need to reorder the outputs as well
            outputs = torch.index_select(outputs, 0, orig_seq_ids)
        return outputs

    def convert_int64_to_int32(self, *args):
        """
        Convert int64 args to int32 to match compiled input types.
        Neuron compiler handles int32 better than int64. Context: P165494809
        """
        return [t.to(torch.int32) if t.dtype == torch.int64 else t for t in args]

    def pad_to_max_compiled_seq(self, *args):
        if self.tag == CONTEXT_ENCODING_MODEL_TAG:
            to_pad = args[:3]
            pad_lengths = [self.neuron_config.max_context_length - arg.shape[1] for arg in to_pad]
            tensor_pad_vals = [self.config.pad_token_id, 0, 1]
            padded_args = [
                F.pad(arg, (0, pad_len), "constant", pad_val)
                for arg, pad_val, pad_len in zip(to_pad, tensor_pad_vals, pad_lengths)
            ]
            args = (*padded_args, *args[3:])
        else:
            input_ids, attention_mask, *rest_of_args = args
            pad_len = self.neuron_config.sequence_length - attention_mask.shape[1]
            padded_attention_mask = F.pad(attention_mask, (0, pad_len), "constant", 0)
            args = (input_ids, padded_attention_mask, *rest_of_args)

        return args

    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        input_ids, attention_mask, position_ids, seq_ids = self.convert_int64_to_int32(
            input_ids, attention_mask, position_ids, seq_ids
        )
        input_ids, attention_mask, position_ids, seq_ids = self.pad_to_max_compiled_seq(
            input_ids, attention_mask, position_ids, seq_ids
        )

        input_batch_size = seq_ids.shape[0]

        if input_batch_size > self.neuron_config.max_batch_size:
            raise ValueError(
                f"Input batch size {input_batch_size} exceeds the maximum batch size {self.neuron_config.max_batch_size}."
            )
        elif input_batch_size == self.neuron_config.batch_size:
            return self._forward(input_ids, attention_mask, position_ids, seq_ids, sampling_params)

        cur_batch = 0
        output_logits = []

        logging.debug(
            f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.neuron_config.batch_size}"
        )

        args = (input_ids, attention_mask, position_ids, seq_ids, sampling_params)
        while cur_batch < input_batch_size:
            if cur_batch + self.neuron_config.batch_size <= input_batch_size:
                # we only process part of the input to run
                logging.debug(f"running foward on batch {cur_batch}:{cur_batch + self.neuron_config.batch_size}")
                outputs = self._forward(*[arg[cur_batch : cur_batch + self.neuron_config.batch_size] for arg in args])
            else:
                # we need to pad the input to run
                logging.debug(
                    f"running forward on batch {cur_batch}:{input_batch_size}, padded up to {self.neuron_config.batch_size}"
                )
                outputs = self._forward_with_pad(*[arg[cur_batch:input_batch_size] for arg in args])

            output_logits.append(outputs)
            cur_batch += self.neuron_config.batch_size

        return torch.cat(output_logits, dim=0)
