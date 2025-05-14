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
import logging
import os
from typing import List

import torch
import torch.nn.functional as F
from neuronx_distributed.trace.model_builder import BaseModelInstance
from torch_neuronx import BucketModelConfig
from transformers import PretrainedConfig

from ...config import NxDNeuronConfig
from ...model_wrapper import NxDModelWrapper
from ..autobucketing import (
    get_context_encoder_bk,
    get_generation_model_bk,
)
from ..generation.sampling import prepare_sampling_params


CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"
SPECULATION_MODEL_TAG = "speculation_model"


def get_bucket_model_config_from_tag(
    tag, config: PretrainedConfig, neuron_config: NxDNeuronConfig, buckets: List[int]
):
    bucket_degree = len(buckets)
    if bucket_degree == 1:
        return None

    pad_token = config.pad_token_id

    # NOTE: KV Cache preprocessing is done within the model and not the
    # shared buffer preprocessor due to lack of support of non-contiguous
    # slicing of nrt tensors via the NRT API.
    if tag == CONTEXT_ENCODING_MODEL_TAG:
        return BucketModelConfig(
            bucket_kernel=get_context_encoder_bk,
            bucket_kernel_constant_args=(
                torch.tensor(buckets),
                neuron_config.padding_side,
                pad_token,
            ),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    elif tag == TOKEN_GENERATION_MODEL_TAG or tag == SPECULATION_MODEL_TAG:
        return BucketModelConfig(
            bucket_kernel=get_generation_model_bk,
            bucket_kernel_constant_args=(
                torch.tensor(buckets),
                neuron_config.padding_side,
                0,
            ),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    else:
        raise ValueError(
            f"The supplied tag: {tag} is not supported for Bucketing. Only {CONTEXT_ENCODING_MODEL_TAG} and {TOKEN_GENERATION_MODEL_TAG} are supported"
        )


class NxDDecoderWrapper(NxDModelWrapper):
    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        buckets: List[int],
        bucket_n_active_tokens: bool,
        model_cls,
        tag="",
        priority_model_idx: int = None,
        model_init_kwargs={},
    ) -> None:
        super().__init__(tag, priority_model_idx)
        self.config = config
        self.neuron_config = neuron_config
        self.buckets = buckets
        self.bucket_n_active_tokens = bucket_n_active_tokens

        if not self.neuron_config.torch_dtype:
            self.neuron_config.torch_dtype = torch.float32

        if config.pad_token_id is None:
            config.pad_token_id = 0

        self.model_cls = model_cls
        self.model = None
        self.is_compiled = False
        self.serialize_base_path = None

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
        self.compiler_workdir = os.path.join(base_compile_work_dir, self.tag)

        self.model_init_kwargs = model_init_kwargs
        self.async_mode = self.neuron_config.async_mode

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.model = self.model_cls(self.config, self.neuron_config)
        self.model.load_state_dict(state_dict, strict=strict, assign=assign)

    def input_generator(
        self,
    ):
        inputs = []
        for bucket in self.buckets:
            n_active_tokens = bucket if self.bucket_n_active_tokens else self.neuron_config.n_active_tokens

            input_ids = torch.zeros((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
            attention_mask = torch.zeros((self.neuron_config.batch_size, bucket), dtype=torch.int32)
            position_ids = torch.zeros((self.neuron_config.batch_size, n_active_tokens), dtype=torch.int32)
            seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)
            # Get the count of sampling params currently supported.
            sampling_params_len = prepare_sampling_params(1).shape[1]
            sampling_params = torch.zeros((self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32)

            inputs.append((input_ids, attention_mask, position_ids, seq_ids, sampling_params))

        return inputs

    def get_model_instance(self):
        return DecoderModelInstance(
            model_cls=self.model_cls,
            config=self.config,
            neuron_config=self.neuron_config,
            buckets=self.buckets,
            **self.model_init_kwargs,
        )

    def get_bucket_config(self):
        return get_bucket_model_config_from_tag(self.tag, self.config, self.neuron_config, self.buckets)

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
        return self.model(input_ids, attention_mask, position_ids, seq_ids, sampling_params)

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

    def _get_async_output(self, ranked_async_tensor):
        outputs = [[async_tensor[0].cpu()] for async_tensor in ranked_async_tensor]
        return outputs[0][0]

    def forward(self, input_ids, attention_mask, position_ids, seq_ids, sampling_params):
        input_ids, attention_mask, position_ids, seq_ids = self.convert_int64_to_int32(
            input_ids, attention_mask, position_ids, seq_ids
        )
        input_ids, attention_mask, position_ids, seq_ids = self.pad_to_max_compiled_seq(
            input_ids, attention_mask, position_ids, seq_ids
        )

        input_batch_size = seq_ids.shape[0]

        if input_batch_size == self.neuron_config.batch_size:
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

        if self.async_mode:
            # block on all requests here, since this is output manipulation
            output_logits = [self._get_async_output(ranked_logits) for ranked_logits in output_logits]

        return torch.cat(output_logits, dim=0)


class DecoderModelInstance(BaseModelInstance):
    def __init__(
        self, model_cls, config: PretrainedConfig, neuron_config: NxDNeuronConfig, buckets: List[int], **kwargs
    ):
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.config = config
        self.neuron_config = neuron_config
        self.buckets = buckets
        self.kwargs = kwargs if kwargs is not None else {}

    def initialize_process_group(self, world_size):
        self.model_cls.initialize_process_group(world_size)

    def load_module(self):
        float_model = self.model_cls(self.config, self.neuron_config, **self.kwargs)
        float_model.eval()

        if self.neuron_config.torch_dtype != torch.float32:
            float_model._apply(
                lambda t: t.to(self.neuron_config.torch_dtype)
                if t.is_floating_point() and t.dtype not in [torch.float8_e4m3fn, torch.float8_e5m2]
                else t
            )
        self.module = float_model

    def get(self, bucket_rank, **kwargs):
        if bucket_rank is not None:
            self.module.n_positions = self.buckets[bucket_rank]

        # Currently we have to init an input_output_aliases map for
        # each buckets, otherwise it will fail the aliasing setup when
        # generating HLO
        self.input_output_aliases = {}
        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2
        # TODO: This else block is a short-term fix for Llava/ViT models to use DecoderModelInstance.
        #       Long-term, these models should use a different implementation of BaseModelInstance.
        if self.module.kv_mgr is not None:
            past_key_values = self.module.kv_mgr.past_key_values
        else:
            past_key_values = self.module.past_key_values
        for i in range(len(past_key_values)):
            self.input_output_aliases[past_key_values[i]] = num_output_from_trace + i
        return self.module, self.input_output_aliases
