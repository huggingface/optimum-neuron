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
# Graph builders used at compilation time to trace decoder models.

import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance
from transformers import PretrainedConfig

from ...config import NxDNeuronConfig
from ...graph_builder import NxDGraphBuilder
from ..generation.sampling import prepare_sampling_params


class NxDDecoderBuilderForCausalLM(NxDGraphBuilder):
    """A graph builder for decoder models used in causal language modeling.

    It supports multiple graphs for:
    - context encoding -> multiple tokens are encoded, and one token is generated
    - token generation -> only one token is encoded and one token is generated
    - speculation -> only one token is encoded and multiple tokens are generated
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        max_tokens: int,
        active_tokens: int,
        model_cls,
        priority_model_idx: int = None,
    ) -> None:
        super().__init__(priority_model_idx)
        self.config = config
        self.neuron_config = neuron_config
        self.max_tokens = max_tokens
        self.active_tokens = active_tokens

        if not self.neuron_config.torch_dtype:
            self.neuron_config.torch_dtype = torch.float32

        if config.pad_token_id is None:
            config.pad_token_id = 0

        self.model_cls = model_cls

    def input_generator(
        self,
    ):
        inputs = []

        input_ids = torch.zeros((self.neuron_config.batch_size, self.active_tokens), dtype=torch.int32)
        attention_mask = torch.zeros((self.neuron_config.batch_size, self.max_tokens), dtype=torch.int32)
        position_ids = torch.zeros((self.neuron_config.batch_size, self.active_tokens), dtype=torch.int32)
        seq_ids = torch.zeros((self.neuron_config.batch_size), dtype=torch.int32)
        # Get the count of sampling params currently supported.
        sampling_params_len = prepare_sampling_params(1).shape[1]
        sampling_params = torch.zeros((self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32)

        inputs.append((input_ids, attention_mask, position_ids, seq_ids, sampling_params))

        return inputs

    def get_model_instance(self):
        return DecoderModelInstanceForCausalLM(
            model_cls=self.model_cls,
            config=self.config,
            neuron_config=self.neuron_config,
            n_positions=self.max_tokens,
        )


class DecoderModelInstanceForCausalLM(BaseModelInstance):
    """Decoder model instance for causal language modeling.

    Aliases the past key values outputs/inputs for faster access during runtime.
    """

    def __init__(self, model_cls, config: PretrainedConfig, neuron_config: NxDNeuronConfig, n_positions: int):
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.config = config
        self.neuron_config = neuron_config
        self.n_positions = n_positions

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
        self.module.n_positions = self.n_positions

    def get(self, bucket_rank, **kwargs):
        assert bucket_rank == 0
        self.input_output_aliases = {}
        num_output_from_trace = 1 if not self.neuron_config.output_logits else 2
        if self.module.kv_mgr is not None:
            past_key_values = self.module.kv_mgr.past_key_values
        else:
            past_key_values = self.module.past_key_values
        for i in range(len(past_key_values)):
            self.input_output_aliases[past_key_values[i]] = num_output_from_trace + i
        return self.module, self.input_output_aliases
