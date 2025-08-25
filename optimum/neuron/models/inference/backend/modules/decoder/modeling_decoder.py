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
import copy
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import neuronx_distributed as nxd
import torch
import torch_xla.core.xla_model as xm
from huggingface_hub import HfApi, snapshot_download
from neuronx_distributed.operators.argmax import argmax as nxd_argmax
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.mappings import (
    _gather_along_dim,
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
)
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from ......cache.entries.single_model import SingleModelCacheEntry
from ......cache.hub_cache import hub_neuronx_cache
from ......modeling_decoder import NeuronModelForCausalLM
from ...config import NxDNeuronConfig
from ...pretrained_model import NxDPreTrainedModel
from ...utils.random import set_random_seed
from ..attention import utils as attn_utils
from ..autobucketing import generate_buckets
from ..flashdecode.utils import (
    get_cache_size,
    mask_util,
    turn_2d_mask_to_4d,
)
from ..generation.generation_utils import NxDGenerationMixin
from ..generation.sampling import (
    Sampler,
    mask_padded_logits,
    prepare_sampling_params,
    validate_sampling_params,
)
from ..kvcache.kv_cache_manager import (
    KVCacheManager,
    _slice_kv_cacheline,
)
from .decoder_wrapper import (
    CONTEXT_ENCODING_MODEL_TAG,
    SPECULATION_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
    NxDDecoderWrapper,
)


logger = logging.getLogger("Neuron")


class NxDDecoderModel(nn.Module):
    """
    Base model that NeuronXXXModel classes inherit from.

    The forward() function will be traced and compiled by NxD.
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
        self.padding_side = neuron_config.padding_side
        self.max_length = neuron_config.sequence_length
        self.sequence_parallel_enabled = neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rank_util = SPMDRank(world_size=neuron_config.tp_degree)
        self.num_cores_per_group = neuron_config.num_cores_per_group
        if neuron_config.on_device_sampling:
            # Instantiate a multinomial Sampler (it can still be used for greedy by passing topk=1)
            self.sampler = Sampler(neuron_config, do_sample=True)
        self.kv_mgr = KVCacheManager(config, neuron_config, num_kv_head=config.num_key_value_heads)

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

    def _create_context_attn_mask(self, attention_mask, **kwargs):
        # Block diagonal causal mask for chunked prefill
        if self.neuron_config.is_chunked_prefill:
            return self._create_chunked_prefill_attn_mask(**kwargs)

        # Lower triangle causal mask for classic attention
        mask = torch.full((self.n_positions, self.n_positions), True, device=attention_mask.device).tril(diagonal=0)
        mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions)

        if self.padding_side == "right":
            return mask
        else:
            expanded_mask = (
                attention_mask[:, None, None, :]
                .expand(self.batch_size, 1, self.n_positions, self.n_positions)
                .to(torch.bool)
            )
            return torch.logical_and(mask, expanded_mask)

    def _create_chunked_prefill_attn_mask(
        self,
        query_lens: torch.Tensor,
        key_lens: torch.Tensor,
        max_query_len: int,
        max_key_len: int,
        **kwargs,
    ) -> torch.Tensor:
        return attn_utils.create_block_diagonal_attn_mask(query_lens, key_lens, max_query_len, max_key_len, **kwargs)

    def _create_spec_attn_mask(self, attention_mask):
        return (
            attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, self.speculation_length, self.n_positions)
            .to(torch.bool)
        )

    def _create_simple_attn_mask(self, attention_mask):
        return attention_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.n_positions).to(torch.bool)

    def create_attn_mask(self, attention_mask, is_for_context_encoding, is_for_speculation, **kwargs):
        if is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask, **kwargs)
        elif is_for_speculation:
            return self._create_spec_attn_mask(attention_mask)
        else:
            return self._create_simple_attn_mask(attention_mask)

    def _slice_kv_cache(self, kv_cache, n_positions):
        past_key_values = []
        for idx in range(len(kv_cache)):
            k_cache = _slice_kv_cacheline(self.neuron_config.padding_side, n_positions, kv_cache[idx][0])
            v_cache = _slice_kv_cacheline(self.neuron_config.padding_side, n_positions, kv_cache[idx][1])
            past_key_values.append([k_cache, v_cache])
        return past_key_values

    def _is_reorder_needed(self, is_for_context_encoding, is_for_speculation):
        return not is_for_context_encoding and not is_for_speculation and self.neuron_config.continuous_batching

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
        scatter_index=None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: torch.FloatTensor | None = None,
        kv_cache: torch.Tensor | None = None,
    ):
        is_for_context_encoding = self._is_context_encoding(input_ids)
        is_for_speculation = self._is_for_speculation(input_ids)

        cache_size = (
            get_cache_size(self.n_positions, self.num_cores_per_group, is_for_context_encoding)
            if self.neuron_config.flash_decoding_enabled
            else self.n_positions
        )

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            if kv_cache is None:
                past_key_values = self.kv_mgr.get_cache(cache_size)
            else:
                past_key_values = self._slice_kv_cache(kv_cache, cache_size)

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask,
            is_for_context_encoding,
            is_for_speculation,
        )
        active_mask = None
        if is_for_speculation:
            active_mask = torch.full(
                (self.speculation_length, self.speculation_length),
                True,
                device=attention_mask.device,
            ).tril(diagonal=0)
            active_mask = active_mask[None, None, :, :].expand(
                self.batch_size, 1, self.speculation_length, self.speculation_length
            )

        # FD masks
        active_mask_2d = None
        if self.neuron_config.flash_decoding_enabled and not is_for_context_encoding:
            rank_id = self.rank_util.get_rank()
            active_mask_2d, attention_mask_2d = mask_util(
                pos_ids=position_ids,
                rank_id=rank_id,
                num_cores_per_group=self.num_cores_per_group,
                cache_size=cache_size,
            )
            active_mask = turn_2d_mask_to_4d(active_mask_2d, n_positions=1, batch_size=self.batch_size)
            attention_mask = turn_2d_mask_to_4d(attention_mask_2d, n_positions=cache_size, batch_size=self.batch_size)

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
            inputs_embeds=inputs_embeds,
        )

        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=past_key_values,
            seq_len=cache_size,
            scatter_index=scatter_index,
            active_mask=active_mask_2d,
            kvcache_buffer=kv_cache,
        )

        batch_size, num_tokens, hidden_size = hidden_states.shape
        if self.padding_side == "left":
            index = torch.tensor([num_tokens - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(batch_size, 1, hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            if not (position_ids.shape[-1] == self.speculation_length or position_ids.shape[-1] == 1):
                # context encoding
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(batch_size, 1, hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if hasattr(self.lm_head, "pad_size"):
            if self.lm_head.gather_output:
                rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
                world_size = 1
            else:
                rank_id = self.rank_util.get_rank()
                world_size = torch.distributed.get_world_size(group=self.lm_head.tensor_parallel_group)
            logits = mask_padded_logits(logits, rank_id, world_size, pad_size=self.lm_head.pad_size)

        res = logits
        if self.neuron_config.on_device_sampling:
            # perform sampling on Neuron to get tokens
            # FIXME, logits[:, -1, :] is not correct for speculation model, this is a tempory fix.
            if is_for_speculation:
                res = nxd_argmax(tensor=logits, dim=2, gather_dim=2, keepdim=False)
            else:
                res = self.sampler(logits[:, -1, :], sampling_params)

        outputs = [res]
        if self.neuron_config.output_logits:
            logits = _gather_along_dim(
                logits,
                partition_dim=2,
            )
            outputs += [logits]
        outputs += updated_kv_cache

        return outputs

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        active_mask: list[torch.FloatTensor] | None = None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: torch.FloatTensor | None = None,
    ):
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  # noqa
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # NeuronLlamaModel class manages the KV cache. So the attention_mask will be generated and passed
        # through to LlamaModel. We override the HF's code that generates attention mask because HF does
        # not support left aligned RHS padding. This enables Neuron to achieve higher performance and
        # extensibility.
        #
        # 4d mask is passed through the layers
        # attention_mask = _prepare_4d_causal_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        # embed positions
        if self.sequence_parallel_enabled:
            # TODO: Replace this with rankid + scatter call once supported
            hidden_states = _reduce_scatter_along_dim(
                inputs_embeds,
                self.sequence_dimension,
                xm.REDUCE_MAX,
            )
        else:
            hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = ()
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
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:]

        hidden_states = self.norm(hidden_states)

        if self.sequence_parallel_enabled:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
            )

        return (hidden_states, next_decoder_cache)


class NxDModelForCausalLM(NxDGenerationMixin, NxDPreTrainedModel, NeuronModelForCausalLM):
    _model_cls = None

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        traced_model: torch.jit.ScriptModule,
        context_encoding_model: NxDDecoderWrapper,
        token_generation_model: NxDDecoderWrapper = None,
        speculation_model: NxDDecoderWrapper = None,
    ):
        self.context_encoding_model = context_encoding_model
        self.token_generation_model = token_generation_model
        self.speculation_model = speculation_model
        # Model wrappers are used by the parent class to assign weights to the model.
        model_wrappers = [self.context_encoding_model]
        if self.token_generation_model is not None:
            model_wrappers.append(self.token_generation_model)
        if self.speculation_model is not None:
            model_wrappers.append(self.speculation_model)
        super().__init__(
            config=config, neuron_config=neuron_config, traced_model=traced_model, model_wrappers=model_wrappers
        )

        self.text_config = self.config.get_text_config()
        self.vocab_size = self.text_config.vocab_size
        self.padding_side = self.neuron_config.padding_side
        self.kv_cache_populated = False

        # async related
        self.async_mode = self.neuron_config.async_mode
        self.next_cpu_inputs = None
        self.prior_outputs = None
        self.unequal_batching = self.neuron_config.ctx_batch_size != self.neuron_config.tkg_batch_size
        if self.async_mode:
            os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "2"

        self.sampler = None

    @staticmethod
    def create_context_encoding_wrapper(model_cls, config, neuron_config, **model_init_kwargs):
        new_neuron_config = copy.deepcopy(neuron_config)
        new_neuron_config.batch_size = neuron_config.ctx_batch_size
        new_neuron_config.n_active_tokens = neuron_config.max_context_length

        if new_neuron_config.enable_bucketing:
            buckets = generate_buckets(128, new_neuron_config.max_context_length)
        else:
            buckets = generate_buckets(
                new_neuron_config.max_context_length,
                new_neuron_config.max_context_length,
            )

        return NxDDecoderWrapper(
            config=config,
            neuron_config=new_neuron_config,
            buckets=buckets,
            bucket_n_active_tokens=True,
            model_cls=model_cls,
            tag=CONTEXT_ENCODING_MODEL_TAG,
            model_init_kwargs=model_init_kwargs,
        )

    @staticmethod
    def create_token_generation_wrapper(
        model_cls, config, neuron_config, enable_wlt_optimization: bool = True, **model_init_kwargs
    ):
        new_neuron_config = copy.deepcopy(neuron_config)
        new_neuron_config.batch_size = neuron_config.tkg_batch_size
        new_neuron_config.n_active_tokens = 1
        new_neuron_config.sequence_parallel_enabled = False

        if new_neuron_config.enable_bucketing:
            buckets = generate_buckets(128, neuron_config.sequence_length)
        else:
            buckets = generate_buckets(neuron_config.sequence_length, neuron_config.sequence_length)

        # shouldn't be used in token gen models
        new_neuron_config.sequence_parallel_enabled = False

        return NxDDecoderWrapper(
            config=config,
            neuron_config=new_neuron_config,
            buckets=buckets,
            bucket_n_active_tokens=False,
            model_cls=model_cls,
            tag=TOKEN_GENERATION_MODEL_TAG,
            priority_model_idx=0 if enable_wlt_optimization else None,  # to turn on weight layout optimization
            model_init_kwargs=model_init_kwargs,
        )

    @staticmethod
    def create_speculation_wrapper(model_cls, config, neuron_config, **model_init_kwargs):
        new_neuron_config = copy.deepcopy(neuron_config)
        new_neuron_config.batch_size = neuron_config.tkg_batch_size
        new_neuron_config.n_active_tokens = neuron_config.speculation_length

        new_neuron_config.sequence_parallel_enabled = False

        if new_neuron_config.enable_bucketing:
            buckets = generate_buckets(128, neuron_config.sequence_length)
        else:
            buckets = generate_buckets(neuron_config.sequence_length, neuron_config.sequence_length)

        return NxDDecoderWrapper(
            config=config,
            neuron_config=new_neuron_config,
            buckets=buckets,
            bucket_n_active_tokens=False,
            model_cls=model_cls,
            tag=SPECULATION_MODEL_TAG,
            priority_model_idx=0,  # to turn on weight layout optimization
            model_init_kwargs=model_init_kwargs,
        )

    @staticmethod
    def create_model_wrappers(model_cls, config, neuron_config, **model_init_kwargs):
        context_encoding_model = NxDModelForCausalLM.create_context_encoding_wrapper(
            model_cls,
            config,
            neuron_config,
            **model_init_kwargs,
        )
        token_generation_model = NxDModelForCausalLM.create_token_generation_wrapper(
            model_cls,
            config,
            neuron_config,
            **model_init_kwargs,
        )
        speculation_model = (
            NxDModelForCausalLM.create_speculation_wrapper(
                model_cls,
                config,
                neuron_config,
                **model_init_kwargs,
            )
            if neuron_config.speculation_length > 0
            else None
        )
        return context_encoding_model, token_generation_model, speculation_model

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
        if self.async_mode:
            # derive future cpu inputs from current cpu inputs
            if position_ids.shape[1] == input_ids.shape[1]:
                next_position_ids = torch.amax(position_ids, 1, keepdim=True)
            else:
                next_position_ids = position_ids

            next_position_ids = next_position_ids + 1
            next_attention_mask = self._infer_attention_mask(next_position_ids)
            self.next_cpu_inputs = {
                "attention_mask": next_attention_mask,
                "position_ids": next_position_ids,
            }

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        if sampling_params is None:
            if self.neuron_config.on_device_sampling:
                raise ValueError("The sampling params tensor is required for on-device sampling.")
            # Just pass a dummy tensor to the model, it will be ignored
            sampling_params = prepare_sampling_params(seq_ids.shape[0])
        elif self.neuron_config.on_device_sampling:
            validate_sampling_params(sampling_params, self.neuron_config.max_topk)

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        logits_or_next_tokens = self._get_model_outputs(
            input_ids,
            attention_mask,
            position_ids,
            seq_ids,
            sampling_params,
        )

        logging.debug("---output---")
        logging.debug(
            f"{'tokens' if self.neuron_config.on_device_sampling else 'logits'} = %s, ",
            logits_or_next_tokens,
        )

        return self._construct_output(logits_or_next_tokens)

    def _setup_func_config(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = output_attentions if output_attentions is not None else self.text_config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.text_config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", None)
        return output_attentions, output_hidden_states, return_dict

    def _infer_attention_mask(self, position_ids):
        assert position_ids is not None, "need to call forward with position_ids if attention_mask is not provided"
        batch_size, seq_len = position_ids.shape
        if position_ids.shape[-1] == 1:
            seq_len = self.neuron_config.sequence_length
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        else:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
        mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
        attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)
        return attention_mask

    def _get_async_output(
        self,
        ranked_async_tensor,
    ):
        outputs = [[async_tensor[0].cpu()] for async_tensor in ranked_async_tensor]
        return outputs[0][0]

    def _get_model_outputs(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        sampling_params,
    ):
        # casting inputs to int32
        input_ids = input_ids.to(torch.int32)
        attention_mask = attention_mask.to(torch.int32)
        position_ids = position_ids.to(torch.int32)
        seq_ids = seq_ids.to(torch.int32)

        if input_ids.shape[-1] > 1 and not position_ids.min().item():
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
            )

            self.kv_cache_populated = True
            if self.async_mode:
                if not self.unequal_batching:
                    # for now only cte + tkg flow is supported with async (this will be enforced at config level)
                    next_outputs = self.token_generation_model(
                        outputs,
                        self.next_cpu_inputs["attention_mask"],
                        self.next_cpu_inputs["position_ids"],
                        seq_ids,
                        sampling_params,
                    )
                    outputs = self._get_async_output(outputs)  # block on cte call
                    self.prior_outputs = next_outputs
                else:
                    if isinstance(
                        outputs, list
                    ):  # in case the outputs weren't passed through `torch.cat` in model_wrapper.py
                        outputs = self._get_async_output(outputs)  # block on cte call

                    self.prior_outputs = None

        elif input_ids.shape[-1] == self.neuron_config.speculation_length:
            outputs = self.speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
            )
        else:
            if (
                self.next_cpu_inputs is not None and self.prior_outputs is not None
            ):  # this is never not None and not in async mode
                _input_ids = self.prior_outputs
                _attention_mask = self.next_cpu_inputs["attention_mask"]
                _position_ids = self.next_cpu_inputs["position_ids"]
            else:
                _input_ids = input_ids
                _attention_mask = attention_mask
                _position_ids = position_ids

            next_outputs = self.token_generation_model(
                _input_ids,
                _attention_mask,
                _position_ids,
                seq_ids,
                sampling_params,
            )
            if self.async_mode:
                if self.prior_outputs is None:  # this means that next_outputs is processing token to be returned
                    self.prior_outputs = next_outputs
                    next_outputs = self.token_generation_model(  # submit future token request
                        next_outputs,
                        self.next_cpu_inputs["attention_mask"],
                        self.next_cpu_inputs["position_ids"],
                        seq_ids,
                        sampling_params,
                    )
                outputs = self.prior_outputs
                if isinstance(outputs, list):
                    outputs = self._get_async_output(
                        self.prior_outputs
                    )  # block on prior (sometimes current) token gen request

                self.prior_outputs = next_outputs
            else:
                outputs = next_outputs

        return outputs

    def _construct_output(self, logits_or_next_tokens):
        next_tokens = logits_or_next_tokens

        OutputParams = CausalLMOutputWithPast(
            logits=None if self.neuron_config.on_device_sampling else logits_or_next_tokens,
            hidden_states=logits_or_next_tokens,
            attentions=None,
        )

        OutputParams.tokens = next_tokens

        return OutputParams

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
        tensorizer_options = (
            "--enable-ccop-compute-overlap "
            f"--cc-pipeline-tiling-factor={neuron_config.cc_pipeline_tiling_factor} "
            "--vectorize-strided-dma "
        )

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

    # NeuronModelForCausalLM methods
    @classmethod
    def _from_pretrained(
        cls,
        model_id: "str | Path",
        config: "PretrainedConfig",
        revision: str | None = None,
        token: bool | str | None = None,
        cache_dir: str | None = None,
        force_download: bool | None = False,
        subfolder: str | None = "",
        local_files_only: bool | None = False,
        trust_remote_code: bool | None = False,
        **kwargs,
    ) -> "NeuronModelForCausalLM":
        if len(kwargs) > 0:
            logger.warning("Ignoring the following kwargs as they are not supported by neuron: %s", kwargs.keys())
        neuron_config = cls.get_neuron_config_cls().from_pretrained(model_id)
        context_encoding_model, token_generation_model, speculation_model = cls.create_model_wrappers(
            model_cls=cls._model_cls,
            config=config,
            neuron_config=neuron_config,
        )
        if not os.path.exists(model_id):
            # The model_id is a model hub id: download the model from the hub.
            with TemporaryDirectory() as tmpdir:
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_dir=tmpdir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    allow_patterns=[cls.COMPILED_MODEL_FILE_NAME],
                )
                traced_model = torch.jit.load(os.path.join(tmpdir, cls.COMPILED_MODEL_FILE_NAME))
        else:
            traced_model = torch.jit.load(os.path.join(model_id, cls.COMPILED_MODEL_FILE_NAME))
        model = cls(
            config=config,
            neuron_config=neuron_config,
            traced_model=traced_model,
            context_encoding_model=context_encoding_model,
            token_generation_model=token_generation_model,
            speculation_model=speculation_model,
        )
        model.load_weights(
            model_id,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
        return model

    @classmethod
    def export(
        cls,
        model_id: str,
        config: "PretrainedConfig | None",
        neuron_config: "NxDNeuronConfig",
        token: bool | str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        force_download: bool | None = False,
        subfolder: str | None = "",
        local_files_only: bool | None = False,
        trust_remote_code: bool | None = False,
        load_weights: bool = True,
        **kwargs,
    ) -> "NeuronModelForCausalLM":
        if len(kwargs) > 0:
            logger.warning("Ignoring the following kwargs as they are not supported by neuron: %s", kwargs.keys())
        if config is None:
            config = AutoConfig.from_pretrained(
                model_id,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                trust_remote_code=trust_remote_code,
            )
        # Override torch_dtype in config as it is used by the neuronx_distributed code to cast weights to the correct type
        config.torch_dtype = neuron_config.torch_dtype
        # Evaluate head_dim if it is defined but set to null (like in Mixtral for transformers 4.54+)
        if hasattr(config, "head_dim") and config.head_dim is None:
            config.head_dim = config.hidden_size // config.num_attention_heads
        context_encoding_model, token_generation_model, speculation_model = cls.create_model_wrappers(
            model_cls=cls._model_cls,
            config=config,
            neuron_config=neuron_config,
        )
        model_wrappers = []
        for wrapper in context_encoding_model, token_generation_model, speculation_model:
            if wrapper is not None:
                model_wrappers.append(wrapper)

        # The model NEFF files will be cached locally, but if the model_id corresponds
        # to a hub model, we also create a cache entry for it.
        cache_entry = (
            None
            if os.path.exists(model_id)
            else SingleModelCacheEntry(model_id, task="text-generation", config=config, neuron_config=neuron_config)
        )
        with hub_neuronx_cache(entry=cache_entry):
            traced_model = NxDPreTrainedModel.compile(
                neuron_config=neuron_config,
                model_wrappers=model_wrappers,
                compiler_args=cls.get_compiler_args(neuron_config),
            )
        model = cls(
            config=config,
            neuron_config=neuron_config,
            traced_model=traced_model,
            context_encoding_model=context_encoding_model,
            token_generation_model=token_generation_model,
            speculation_model=speculation_model,
        )
        if load_weights:
            model.load_weights(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )
        return model

    def _save_pretrained(self, save_directory: str | Path):
        model_name_or_path = getattr(self.config, "_name_or_path")
        # If the model was exported from a local path, we need to save the checkpoint (not that we also shard it)
        weight_path = model_name_or_path if os.path.isdir(model_name_or_path) else None
        self.save(save_directory, weight_path=weight_path)

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: bool | None = None,
        revision: str | None = None,
        token: bool | str = True,
        endpoint: str | None = None,
    ) -> str:
        api = HfApi(endpoint=endpoint)

        api.create_repo(
            token=token,
            repo_id=repository_id,
            exist_ok=True,
            private=private,
        )
        ignore_patterns = []
        checkpoint_id = self.neuron_config.checkpoint_id
        if checkpoint_id is not None:
            # Avoid uploading checkpoints when the original model is available on the hub
            ignore_patterns = [self.CHECKPOINT_DIR + "/*"]
        api.upload_folder(
            repo_id=repository_id,
            folder_path=save_directory,
            token=token,
            revision=revision,
            ignore_patterns=ignore_patterns,
        )
