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
# Task-specific decoder model implementations for inference on AWS Neuron.
import copy
import logging

import neuronx_distributed as nxd
import torch
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_dim,
)
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ....modeling_utils import NeuronModelForCausalLM, NeuronModelForEmbedding
from ...config import NxDNeuronConfig
from ...graph_builder import NxDGraphBuilder
from ...pretrained_model import NxDPreTrainedModel
from ...utils.random import set_random_seed
from ..attention.gqa import get_shardable_head_counts
from ..generation.generation_utils import NxDGenerationMixin
from ..generation.sampling import (
    Sampler,
    mask_padded_logits,
    validate_sampling_params,
)
from ..kvcache.kv_cache_manager import (
    KVCacheManager,
)
from .decoder_builders import NxDDecoderBuilderForCausalLM, NxDDecoderBuilderForEmbedding
from .decoder_wrappers import (
    CONTEXT_ENCODING_MODEL_TAG,
    SPECULATION_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    NxDDecoderWrapperForCausalLM,
    NxDDecoderWrapperForEmbedding,
)


logger = logging.getLogger("Neuron")


class NxDDecoderModelForCausalLM(nn.Module):
    """A decoder model used for causal language modeling.

    The forward() function will be traced and compiled for different use cases:
    - context encoding -> multiple tokens are encoded, and one token is generated
    - token generation -> only one token is encoded and one token is generated
    - speculation -> only one token is encoded and multiple tokens are generated

    The model manages its own KV cache and sampling logic. It supports on-device sampling.
    and continuous batching.
    """

    def __init__(self, config: PretrainedConfig, neuron_config: NxDNeuronConfig):
        super().__init__()

        self.config = config
        self.sampler = None
        self.kv_mgr = None
        self.neuron_config = neuron_config
        self.batch_size = neuron_config.batch_size
        self.n_positions = neuron_config.sequence_length
        self.vocab_size = config.vocab_size
        self.speculation_length = neuron_config.speculation_length
        self.max_length = neuron_config.sequence_length
        self.rank_util = SPMDRank(world_size=neuron_config.tp_degree)
        if neuron_config.on_device_sampling:
            # Instantiate a multinomial Sampler (it can still be used for greedy by passing topk=1)
            self.sampler = Sampler(neuron_config, do_sample=True)
        # Evaluate the sharding strategy and number of kv heads per rank
        _, num_attention_heads, num_key_value_heads = get_shardable_head_counts(
            neuron_config.tp_degree, config.num_attention_heads, config.num_key_value_heads
        )
        if num_attention_heads != config.num_attention_heads:
            logger.info(
                f"Adjusting num_attention_heads from {config.num_attention_heads} to {num_attention_heads} for TP {neuron_config.tp_degree}."
            )
        if num_key_value_heads != config.num_key_value_heads:
            logger.info(
                f"Adjusting num_key_value_heads from {config.num_key_value_heads} to {num_key_value_heads} for TP {neuron_config.tp_degree}."
            )
        self.kv_mgr = KVCacheManager(config, neuron_config, actual_num_key_value_heads=num_key_value_heads)

    def initialize_process_group(self, seed: int = 0):
        if not torch.dist.is_initialized():
            torch.dist.init_process_group(backend="xla")
        else:
            logging.warning("torch.distributed was already initialized, skipping...")

        if not nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            nxd.parallel_layers.initialize_model_parallel(
                tensor_model_parallel_size=self.neuron_config.tp_degree,
                pipeline_model_parallel_size=self.neuron_config.pp_degree,
                expert_model_parallel_size=self.neuron_config.ep_degree,
            )
        else:
            logging.warning("NxD was already initialized, skipping...")

        # set seed
        set_random_seed(seed)

    def _is_context_encoding(self, input_ids: torch.Tensor):
        return input_ids.shape[-1] > 1 and input_ids.shape[-1] != self.speculation_length

    def _is_for_speculation(self, input_ids: torch.Tensor):
        return input_ids.shape[-1] == self.speculation_length

    def forward(
        self,
        input_ids,
        position_ids,
        seq_ids,
        sampling_params,
    ):
        """Forward pass that can return either logits or hidden states.

        Args:
            input_ids (torch.LongTensor): Input token IDs.
            position_ids (torch.LongTensor): Position IDs.
            seq_ids (torch.LongTensor): Sequence IDs. Used in continuous batching
            sampling_params (torch.FloatTensor): Sampling parameters.
        """
        is_for_context_encoding = self._is_context_encoding(input_ids)
        is_for_speculation = self._is_for_speculation(input_ids)

        cache_size = self.n_positions

        # Prepare input tensors
        device = input_ids.device
        if is_for_context_encoding:
            past_key_values = None
            # Lower triangle causal mask for classic attention
            # Note that the mask is created for the full sequence length even if only a part of it is used
            attention_mask = torch.full((self.n_positions, self.n_positions), True, device=device).tril(diagonal=0)
            attention_mask = attention_mask[None, None, :, :].expand(
                self.batch_size, 1, self.n_positions, self.n_positions
            )
            active_mask = None
        else:
            past_key_values = self.kv_mgr.get_cache(cache_size)
            # Prepare attention mask(s) as expected by the decoding/speculation models
            # The full attention mask is split into two parts:
            # - the attention_mask for the cached tokens,
            # - the active_mask for the newly generated token(s)
            if is_for_speculation:
                # For speculation, the index of the last cached token is the first position id minus one
                max_cached_positions = position_ids[:, :1].expand(self.batch_size, self.n_positions) - 1
            else:
                # For decoding, the index of the last cached token is the (only) position id minus one
                max_cached_positions = position_ids.expand(self.batch_size, self.n_positions) - 1
            all_positions = (
                torch.arange(self.n_positions, device=device).view(1, -1).expand(self.batch_size, self.n_positions)
            )
            attention_mask = (max_cached_positions >= all_positions).view(self.batch_size, 1, 1, self.n_positions)
            if is_for_speculation:
                attention_mask = attention_mask.expand(self.batch_size, 1, self.speculation_length, self.n_positions)
                active_mask = torch.full(
                    (self.speculation_length, self.speculation_length),
                    True,
                    device=attention_mask.device,
                ).tril(diagonal=0)
                active_mask = active_mask[None, None, :, :].expand(
                    self.batch_size, 1, self.speculation_length, self.speculation_length
                )
            else:
                # Active mask is implicit for decoding
                active_mask = None

        batch_size, seq_length = input_ids.shape[:2]
        position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # decoder layers
        new_key_values = []
        cos_cache = None
        sin_cache = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )

            hidden_states = layer_outputs[0]
            new_key_values.append(layer_outputs[1])
            cos_cache, sin_cache = layer_outputs[2:]

        hidden_states = self.norm(hidden_states)

        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=new_key_values,
            seq_len=cache_size,
        )

        hidden_size = hidden_states.shape[-1]
        if is_for_context_encoding:
            # Do not evaluate logits for all tokens in the sequence, only the last one
            index = torch.max(position_ids, dim=1, keepdim=True).indices
            index = index.unsqueeze(1).expand(batch_size, 1, hidden_size)
            # (batch_size, seq_length, hidden_size) -> (batch_size, 1, hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.lm_head.gather_output:
            # The lm_head outputs are already gathered on rank 0
            rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
            world_size = 1
        else:
            # The lm_head outputs are sharded across tensor parallel ranks
            # This is usually the case when on-device-sampling is used
            rank_id = self.rank_util.get_rank()
            world_size = torch.distributed.get_world_size(group=self.lm_head.tensor_parallel_group)

        if hasattr(self.lm_head, "pad_size"):
            logits = mask_padded_logits(logits, rank_id, world_size, pad_size=self.lm_head.pad_size)

        res = logits
        if self.neuron_config.on_device_sampling:
            # perform sampling on Neuron to get tokens
            # FIXME, logits[:, -1, :] is not correct for speculation model, this is a tempory fix.
            if is_for_speculation:
                res = nxd_argmax(tensor=logits, dim=2, gather_dim=2, keepdim=False)
            else:
                res = self.sampler(logits[:, -1, :], sampling_params, rank_id=rank_id)

        outputs = [res]
        if self.neuron_config.output_logits:
            logits = _gather_along_dim(
                logits,
                partition_dim=2,
            )
            outputs += [logits]
        outputs += updated_kv_cache

        return outputs


class NxDModelForCausalLM(NxDGenerationMixin, NxDPreTrainedModel, NeuronModelForCausalLM):
    """Base class for neuron causal language modeling.

    It uses separate model graphs for context encoding, token generation and speculation.

    """

    _model_cls = None

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        traced_model: torch.jit.ScriptModule,
        graph_builders: list[NxDGraphBuilder],
    ):
        super().__init__(
            config=config, neuron_config=neuron_config, traced_model=traced_model, graph_builders=graph_builders
        )
        ctx_neuron_config = NxDModelForCausalLM._create_context_encoding_config(neuron_config)
        self.context_encoding_model = NxDDecoderWrapperForCausalLM(
            config=config, neuron_config=ctx_neuron_config, model=traced_model, tag=CONTEXT_ENCODING_MODEL_TAG
        )
        tkg_neuron_config = NxDModelForCausalLM._create_token_generation_config(neuron_config)
        self.token_generation_model = NxDDecoderWrapperForCausalLM(
            config=config, neuron_config=tkg_neuron_config, model=traced_model, tag=TOKEN_GENERATION_MODEL_TAG
        )
        if neuron_config.speculation_length > 0:
            spec_neuron_config = NxDModelForCausalLM._create_speculation_config(neuron_config)
            self.speculation_model = NxDDecoderWrapperForCausalLM(
                config=config,
                neuron_config=spec_neuron_config,
                model=traced_model,
                tag=SPECULATION_MODEL_TAG,
            )

        self.text_config = self.config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.kv_cache_populated = False
        self.sampler = None

    @staticmethod
    def _create_context_encoding_config(neuron_config: NxDNeuronConfig) -> NxDNeuronConfig:
        ctx_neuron_config = copy.deepcopy(neuron_config)
        ctx_neuron_config.batch_size = neuron_config.ctx_batch_size
        return ctx_neuron_config

    @staticmethod
    def _create_token_generation_config(neuron_config: NxDNeuronConfig) -> NxDNeuronConfig:
        tkg_neuron_config = copy.deepcopy(neuron_config)
        tkg_neuron_config.batch_size = neuron_config.tkg_batch_size
        return tkg_neuron_config

    @staticmethod
    def _create_speculation_config(neuron_config: NxDNeuronConfig) -> NxDNeuronConfig:
        spec_neuron_config = copy.deepcopy(neuron_config)
        spec_neuron_config.batch_size = neuron_config.tkg_batch_size
        return spec_neuron_config

    @classmethod
    def create_graph_builders(cls, config, neuron_config):
        if cls._model_cls is None:
            raise SystemError(f"No underlying model class defined for {cls}.")
        graph_builders = {}
        ctx_neuron_config = NxDModelForCausalLM._create_context_encoding_config(neuron_config)
        graph_builders["context_encoding"] = NxDDecoderBuilderForCausalLM(
            config=config,
            neuron_config=ctx_neuron_config,
            max_tokens=ctx_neuron_config.max_context_length,
            active_tokens=ctx_neuron_config.max_context_length,
            model_cls=cls._model_cls,
        )
        tkg_neuron_config = NxDModelForCausalLM._create_token_generation_config(neuron_config)
        graph_builders["token_generation"] = NxDDecoderBuilderForCausalLM(
            config=config,
            neuron_config=tkg_neuron_config,
            max_tokens=tkg_neuron_config.sequence_length,
            active_tokens=1,
            model_cls=cls._model_cls,
            priority_model_idx=0,  # to turn on weight layout optimization
        )
        if neuron_config.speculation_length > 0:
            spec_neuron_config = NxDModelForCausalLM._create_speculation_config(neuron_config)
            graph_builders["speculation_model"] = NxDDecoderBuilderForCausalLM(
                config=config,
                neuron_config=spec_neuron_config,
                max_tokens=spec_neuron_config.sequence_length,
                active_tokens=spec_neuron_config.speculation_length,
                model_cls=cls._model_cls,
                priority_model_idx=0,  # to turn on weight layout optimization
            )
        return graph_builders

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor | None,
        seq_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        sampling_params: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        if output_attentions:
            raise ValueError(f"output_attentions is not supported for {self.__class__.__name__}")
        if output_hidden_states:
            raise ValueError(f"output_hidden_states is not supported for {self.__class__.__name__}")
        if return_dict:
            raise ValueError(f"return_dict is not supported for {self.__class__.__name__}")
        if self.neuron_config.on_device_sampling:
            validate_sampling_params(sampling_params, self.neuron_config.max_topk)

        if input_ids.shape[-1] > 1 and not position_ids.min().item():
            outputs = self.context_encoding_model(
                input_ids,
                position_ids,
                seq_ids,
                sampling_params,
            )

            self.kv_cache_populated = True

        elif input_ids.shape[-1] == self.neuron_config.speculation_length:
            outputs = self.speculation_model(
                input_ids,
                position_ids,
                seq_ids,
                sampling_params,
            )
        else:
            outputs = self.token_generation_model(
                input_ids,
                position_ids,
                seq_ids,
                sampling_params,
            )

        return outputs

    def reset(self):
        # We need to reset the KV cache flag for a new batch of inference.
        # When the flag is reset, the subsequent run will invoke the
        # context encoding model.
        self.kv_cache_populated = False

    def get_required_kwargs(self) -> list[str]:
        """The list of required kwargs to the model's forward"""
        return []

    @classmethod
    def get_compiler_args(cls, neuron_config: NxDNeuronConfig) -> str:
        tensorizer_options = "--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma "

        compiler_args = (
            "--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='{tensorizer_options}'"
            " -O2 "
            f" --lnc={neuron_config.logical_nc_config}"
        )

        if neuron_config.target:
            compiler_args += f" --target {neuron_config.target}"

        logging.info(f"neuronx-cc compiler_args are: {compiler_args}")
        return compiler_args


class NxDDecoderModelForEmbedding(nn.Module):
    """A decoder model used for text embedding and ranking tasks.

    It uses a single model graph for encoding the input text and extracting the embeddings.
    """

    def __init__(self, config: PretrainedConfig, neuron_config: NxDNeuronConfig):
        super().__init__()

        self.config = config
        self.neuron_config = neuron_config
        self.batch_size = neuron_config.batch_size
        self.n_positions = neuron_config.sequence_length
        self.vocab_size = config.vocab_size
        self.max_length = neuron_config.sequence_length
        self.rank_util = SPMDRank(world_size=neuron_config.tp_degree)

    def initialize_process_group(self, seed: int = 0):
        if not torch.dist.is_initialized():
            torch.dist.init_process_group(backend="xla")
        else:
            logging.warning("torch.distributed was already initialized, skipping...")

        if not nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            nxd.parallel_layers.initialize_model_parallel(
                tensor_model_parallel_size=self.neuron_config.tp_degree,
                pipeline_model_parallel_size=self.neuron_config.pp_degree,
                expert_model_parallel_size=self.neuron_config.ep_degree,
            )
        else:
            logging.warning("NxD was already initialized, skipping...")

        # set seed
        set_random_seed(seed)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        # Prepare attention mask(s)
        attention_mask = torch.full((self.n_positions, self.n_positions), True, device=attention_mask.device).tril(
            diagonal=0
        )
        attention_mask = attention_mask[None, None, :, :].expand(
            self.batch_size, 1, self.n_positions, self.n_positions
        )

        hidden_states = self.embed_tokens(input_ids)

        cos_cache = None
        sin_cache = None
        for decoder_layer in self.layers:
            hidden_states, _, cos_cache, sin_cache = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )

        hidden_states = self.norm(hidden_states)

        batch_size, _, hidden_size = hidden_states.shape
        index = torch.max(position_ids, dim=1, keepdim=True).indices
        index = index.unsqueeze(1).expand(batch_size, 1, hidden_size)
        hidden_states = torch.gather(hidden_states, dim=1, index=index)

        return hidden_states


class NxDModelForEmbedding(NxDPreTrainedModel, NeuronModelForEmbedding):
    """Base class for neuron embeddings."""

    _model_cls = None

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        traced_model: torch.jit.ScriptModule,
        graph_builders: list[NxDGraphBuilder],
    ):
        super().__init__(
            config=config, neuron_config=neuron_config, traced_model=traced_model, graph_builders=graph_builders
        )
        self.encoding_model = NxDDecoderWrapperForEmbedding(
            config=config,
            neuron_config=neuron_config,
            model=traced_model,
        )
        self.text_config = self.config.get_text_config()
        self.vocab_size = self.text_config.vocab_size

    @classmethod
    def create_graph_builders(cls, config, neuron_config):
        if cls._model_cls is None:
            raise SystemError(f"No underlying model class defined for {cls}.")
        return {
            "encoding": NxDDecoderBuilderForEmbedding(
                config=config,
                neuron_config=neuron_config,
                max_tokens=neuron_config.max_context_length,
                model_cls=cls._model_cls,
            )
        }

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        batch_size, seq_len = input_ids.shape
        # Create position_ids
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        input_ids = input_ids.to(torch.int32)
        attention_mask = attention_mask.to(torch.int32)
        position_ids = position_ids.to(torch.int32)
        return self.encoding_model(
            input_ids,
            attention_mask,
            position_ids,
        )

    @classmethod
    def get_compiler_args(cls, neuron_config: NxDNeuronConfig) -> str:
        tensorizer_options = "--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2 --vectorize-strided-dma "

        compiler_args = (
            "--auto-cast=none --model-type=transformer "
            f"--tensorizer-options='{tensorizer_options}'"
            " -O2 "
            f" --lnc={neuron_config.logical_nc_config}"
        )

        if neuron_config.target:
            compiler_args += f" --target {neuron_config.target}"

        logging.info(f"neuronx-cc compiler_args are: {compiler_args}")
        return compiler_args
