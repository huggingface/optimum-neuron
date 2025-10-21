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

import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance
from torch_neuronx import BucketModelConfig
from transformers import PretrainedConfig

from ...config import NxDNeuronConfig
from ...graph_builder import NxDGraphBuilder
from ..autobucketing import (
    get_context_encoder_bk,
    get_generation_model_bk,
)
from ..generation.sampling import prepare_sampling_params


CONTEXT_ENCODING_MODEL_TAG = "context_encoding_model"
TOKEN_GENERATION_MODEL_TAG = "token_generation_model"
SPECULATION_MODEL_TAG = "speculation_model"


def get_bucket_model_config_from_tag(
    tag, config: PretrainedConfig, neuron_config: NxDNeuronConfig, buckets: list[int]
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
                0,
            ),
            shared_state_buffer=None,
            func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
        )
    else:
        raise ValueError(
            f"The supplied tag: {tag} is not supported for Bucketing. Only {CONTEXT_ENCODING_MODEL_TAG} and {TOKEN_GENERATION_MODEL_TAG} are supported"
        )


class NxDDecoderBuilder(NxDGraphBuilder):
    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        buckets: list[int],
        bucket_n_active_tokens: bool,
        model_cls,
        tag="",
        priority_model_idx: int = None,
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
        )

    def get_bucket_config(self):
        return get_bucket_model_config_from_tag(self.tag, self.config, self.neuron_config, self.buckets)


class DecoderModelInstance(BaseModelInstance):
    def __init__(self, model_cls, config: PretrainedConfig, neuron_config: NxDNeuronConfig, buckets: list[int]):
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.config = config
        self.neuron_config = neuron_config
        self.buckets = buckets

    def initialize_process_group(self, world_size):
        self.model_cls.initialize_process_group(world_size)

    def load_module(self):
        float_model = self.model_cls(self.config, self.neuron_config)
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
