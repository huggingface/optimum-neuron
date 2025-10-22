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

from ....modeling_utils import NeuronModelForCausalLM
from ...config import NxDNeuronConfig
from ...graph_builder import NxDGraphBuilder
from ...pretrained_model import NxDPreTrainedModel
from ...utils.random import set_random_seed
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
from .decoder_builder import NxDDecoderBuilder
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
        self.max_length = neuron_config.sequence_length
        self.rank_util = SPMDRank(world_size=neuron_config.tp_degree)
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
        # Lower triangle causal mask for classic attention
        mask = torch.full((self.n_positions, self.n_positions), True, device=attention_mask.device).tril(diagonal=0)
        mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions)
        return mask

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
            k_cache = _slice_kv_cacheline(n_positions, kv_cache[idx][0])
            v_cache = _slice_kv_cacheline(n_positions, kv_cache[idx][1])
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

        cache_size = self.n_positions

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
        if not (position_ids.shape[-1] == self.speculation_length or position_ids.shape[-1] == 1):
            # context encoding
            index = torch.max(position_ids, dim=1, keepdim=True).indices
            index = index.unsqueeze(1).expand(batch_size, 1, hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # rank_id and world_size are required for padding and sampling
        # TODO: check if both code paths below are used and when
        if self.lm_head.gather_output:
            rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
            world_size = 1
        else:
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

        return (hidden_states, next_decoder_cache)


class NxDModelForCausalLM(NxDGenerationMixin, NxDPreTrainedModel, NeuronModelForCausalLM):
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
        self.context_encoding_model = NxDDecoderWrapper(
            config=config, neuron_config=ctx_neuron_config, model=traced_model, tag=CONTEXT_ENCODING_MODEL_TAG
        )
        tkg_neuron_config = NxDModelForCausalLM._create_token_generation_config(neuron_config)
        self.token_generation_model = NxDDecoderWrapper(
            config=config, neuron_config=tkg_neuron_config, model=traced_model, tag=TOKEN_GENERATION_MODEL_TAG
        )
        if neuron_config.speculation_length > 0:
            spec_neuron_config = NxDModelForCausalLM._create_speculation_config(neuron_config)
            self.speculation_model = NxDDecoderWrapper(
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
        graph_builders["context_encoding"] = NxDDecoderBuilder(
            config=config,
            neuron_config=ctx_neuron_config,
            max_tokens=ctx_neuron_config.max_context_length,
            active_tokens=ctx_neuron_config.max_context_length,
            model_cls=cls._model_cls,
        )
        tkg_neuron_config = NxDModelForCausalLM._create_token_generation_config(neuron_config)
        graph_builders["token_generation"] = NxDDecoderBuilder(
            config=config,
            neuron_config=tkg_neuron_config,
            max_tokens=tkg_neuron_config.sequence_length,
            active_tokens=1,
            model_cls=cls._model_cls,
            priority_model_idx=0,  # to turn on weight layout optimization
        )
        if neuron_config.speculation_length > 0:
            spec_neuron_config = NxDModelForCausalLM._create_speculation_config(neuron_config)
            graph_builders["speculation_model"] = NxDDecoderBuilder(
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

        elif input_ids.shape[-1] == self.neuron_config.speculation_length:
            outputs = self.speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
            )
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                sampling_params,
            )

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
