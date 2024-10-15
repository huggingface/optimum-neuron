import os
import copy
import tempfile
import warnings

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
from torch import nn
from modules.autobucketing import generate_buckets
from modules.checkpoint import load_state_dict
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from neuronx_distributed.quantization.quantization_config import QuantizationType
from safetensors.torch import load_file

from neuronx_distributed.quantization.quantization_utils import (
    convert_qint8_to_int8_state_dict,
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)
from neuronx_distributed.parallel_layers import parallel_state, utils  # noqa: E402
from neuronx_distributed.trace.model_builder import ModelBuilder
from neuronx_distributed.utils.speculative_decoding import NeuronSpeculation
from neuronx_distributed.utils.sampling import Sampler  # noqa: E402

from modules.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,  # noqa: E402
    SPECULATION_MODEL_TAG,  # noqa: E402
    MEDUSA_MODEL_TAG,  # noqa: E402
    TOKEN_GENERATION_MODEL_TAG,  # noqa: E402
    ModelWrapper,  # noqa: E402
)

SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

from modules.autobucketing import slice_lhs, slice_rhs  # noqa: E402
from modules.gqa import (  # noqa: E402
    determine_sharding_strategy,  # noqa: E402
    get_shardable_head_counts,  # noqa: E402
)  # noqa: E402


class NeuronBaseModel(PreTrainedModel):
    """
    Base model that NeuronXXXModel classes inherit from.

    The forward() function will be traced and compiled by NxD.
    """

    SEQ_DIM = 2

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.batch_size = config.batch_size
        self.n_positions = config.n_positions
        self.vocab_size = config.vocab_size
        self.speculation_length = config.speculation_length
        self.padding_side = config.padding_side
        self.max_length = config.max_length

        self.setup_attr_for_model(config)
        self.init_model(config)
        self.init_inference_optimization(config)
        self.post_init()

    def setup_attr_for_model(self, config: PretrainedConfig):
        """
        Please provide model-specific definition for the following attributes
            self.on_device_sampling
            self.tp_degree
            self.hidden_size
            self.num_attention_heads
            self.num_key_value_heads
            self.max_batch_size
            self.buckets
        """
        raise NotImplementedError("setup_attr_for_model() is not implemented")

    def init_model(self, config: PretrainedConfig):
        """
        Please provide definition for the following components:
            self.embed_tokens
            self.layers
            self.norm
            self.lm_head
        """
        raise NotImplementedError("init_model() is not implemented")

    def init_inference_optimization(self, config: PretrainedConfig):
        if self.on_device_sampling:
            self.sampler = Sampler(config)

        gqa_sharding_strategy = determine_sharding_strategy(self.tp_degree, self.num_key_value_heads)
        _, num_key_value_heads = get_shardable_head_counts(
            self.tp_degree, self.num_attention_heads, self.num_key_value_heads, gqa_sharding_strategy
        )
        if parallel_state.model_parallel_is_initialized():
            num_kv_heads_per_partition = utils.divide(num_key_value_heads, self.tp_degree)
        else:
            num_kv_heads_per_partition = num_key_value_heads

        hidden_dim_per_head = self.hidden_size // self.num_attention_heads

        self.kv_shape = (
            self.max_batch_size,
            num_kv_heads_per_partition,
            self.max_length,
            hidden_dim_per_head,
        )
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.kv_shape, dtype=config.torch_dtype), requires_grad=False)
                for _ in range(config.num_hidden_layers * 2)
            ]
        )

    def _bucket_slice_kv_cacheline(self, cache):

        if self.padding_side == "right":
            return slice_lhs(cache, self.n_positions, self.SEQ_DIM)
        else:
            max_idx = cache.shape[self.SEQ_DIM]
            return slice_rhs(cache, self.n_positions, max_idx, self.SEQ_DIM)

    def _gather_bucket_slice_into_kv_cacheline(self, idx, bucket_slice):
        max_idx = self.past_key_values[idx].shape[self.SEQ_DIM]
        if self.padding_side == "right":
            remaining = slice_rhs(self.past_key_values[idx], max_idx - self.n_positions, max_idx, self.SEQ_DIM)
            return torch.cat([bucket_slice, remaining], dim=self.SEQ_DIM)
        else:
            remaining = slice_lhs(self.past_key_values[idx], max_idx - self.n_positions, self.SEQ_DIM)
            return torch.cat([remaining, bucket_slice], dim=self.SEQ_DIM)

    def _create_context_attn_mask(self, attention_mask):
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

    def _create_spec_attn_mask(self, attention_mask):
        return (
            attention_mask[:, None, None, :]
            .expand(self.batch_size, 1, self.speculation_length, self.n_positions)
            .to(torch.bool)
        )

    def _create_simple_attn_mask(self, attention_mask):
        return attention_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.n_positions).to(torch.bool)

    def create_attn_mask(self, attention_mask, is_for_context_encoding, is_for_speculation, position_ids):
        if is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask)
        elif is_for_speculation:
            return self._create_spec_attn_mask(attention_mask)
        else:
            return self._create_simple_attn_mask(attention_mask)

    def _medusa_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        accepted_indices = None,
        current_length = None,
        medusa_mask = None,
        scatter_index = None,
    ):
        is_for_context_encoding = (
            input_ids.shape[-1] > 1
            and self.medusa_speculation_length != input_ids.shape[-1]
        )
        is_for_medusa_speculation = input_ids.shape[-1] == self.medusa_speculation_length

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            past_key_values = []
            if is_for_medusa_speculation:
                index = current_length.view(-1, 1, current_length.shape[-1], 1).expand_as(
                    self.past_key_values[0][:, :, 0 : self.config.num_medusa_heads + 1, :]
                )
                gather_index = accepted_indices.view(-1, 1, accepted_indices.shape[-1], 1).expand_as(
                    self.past_key_values[0][:, :, 0 : self.config.num_medusa_heads + 1, :]
                )

                for key_layer_idx in range(0, len(self.past_key_values), 2):
                    k_cache = self.past_key_values[key_layer_idx]
                    v_cache = self.past_key_values[key_layer_idx + 1]

                    accepted_k_cache = torch.gather(k_cache, dim=2, index=gather_index)
                    accepted_v_cache = torch.gather(v_cache, dim=2, index=gather_index)
                    k_cache = torch.scatter(k_cache, 2, index, accepted_k_cache)
                    v_cache = torch.scatter(v_cache, 2, index, accepted_v_cache)

                    key_state = self._bucket_slice_kv_cacheline(k_cache)
                    value_state = self._bucket_slice_kv_cacheline(v_cache)

                    past_key_values.append([key_state, value_state])

            else:
                for key_layer_idx in range(0, len(self.past_key_values), 2):
                    k_cache = self.past_key_values[key_layer_idx]
                    v_cache = self.past_key_values[key_layer_idx + 1]
                    key_state = self._bucket_slice_kv_cacheline(k_cache)
                    value_state = self._bucket_slice_kv_cacheline(v_cache)

                    past_key_values.append([key_state, value_state])

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask, is_for_context_encoding, False, position_ids
        )
        active_mask = None
        if is_for_medusa_speculation:
            medusa_mask = medusa_mask[0].bool()
            active_mask = medusa_mask[None, None, :, :].expand(
                self.batch_size, 1, self.medusa_speculation_length, self.medusa_speculation_length
            )

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
        )

        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
            k_cache = self.past_key_values[idx * 2]
            v_cache = self.past_key_values[idx * 2 + 1]

            if is_for_context_encoding:
                if self.config.is_continuous_batching:
                    # scatter back to the desired seq_ids
                    seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)
                    k_cache = torch.scatter(k_cache, 0, seq_id_index, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 0, seq_id_index, kv_per_layer[1])
                else:
                    # assign back to full kv_cacheline
                    k_cache = kv_per_layer[0]
                    v_cache = kv_per_layer[1]
                    k_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2, k_cache)
                    v_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2 + 1, v_cache)
            else:
                if self.padding_side == "left":
                    # TODO: fix it with scatter after right padding
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, kv_per_layer[0]], dim=2)
                    v_cache = torch.cat([v_cache, kv_per_layer[1]], dim=2)
                else:
                    if is_for_medusa_speculation:
                        scatter_index_new = scatter_index.view(-1, 1, scatter_index.shape[-1], 1).expand_as(
                            kv_per_layer[0]
                        )
                    else:
                        scatter_index_new = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(
                            kv_per_layer[0]
                        )
                    k_cache = torch.scatter(k_cache, 2, scatter_index_new, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 2, scatter_index_new, kv_per_layer[1])

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            if position_ids.shape[-1] == self.medusa_speculation_length:
                index = torch.min(position_ids)
                index = torch.arange(index, index + self.medusa_speculation_length, device=hidden_states.device)
                index = index[None, :, None].expand(
                    self.batch_size, self.medusa_speculation_length, self.hidden_size
                )
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                # simple token generation
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        medusa_logits = [logits] + [
            head(hidden_states).float()
            for head in [getattr(self, f"medusa_head_{i}") for i in range(self.num_medusa_heads)]
        ]
        stacked_logits = torch.stack(medusa_logits, dim=0)

        res = logits
        if is_for_context_encoding:
            result = [
                self.sampler.sample(stacked_logits[i : i + 1, -1, :].squeeze(0))
                for i in range(self.config.num_medusa_heads + 1)
            ]
            res = torch.stack(result, dim=0)  # 5, 1, 10
        else:
            results = []
            for i in range(stacked_logits.shape[1]):
                result = [
                    self.sampler.sample(stacked_logits[j : j + 1, i, :].squeeze(0))
                    for j in range(self.config.num_medusa_heads + 1)
                ]
                res = torch.stack(result, dim=0)
                results.append(res)

        return [res] + updated_kv_cache

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        accepted_indices = None,
        current_length = None,
        medusa_mask = None,
        scatter_index = None,
    ):
        if self.config.is_medusa:
            return self._medusa_forward(input_ids, attention_mask, position_ids, seq_ids, accepted_indices, current_length, medusa_mask, scatter_index)

        is_for_context_encoding = (
            input_ids.shape[-1] > 1
            and self.speculation_length != input_ids.shape[-1]
        )
        is_for_speculation = input_ids.shape[-1] == self.speculation_length

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            past_key_values = []
            for key_layer_idx in range(0, len(self.past_key_values), 2):
                k_cache = self.past_key_values[key_layer_idx]
                v_cache = self.past_key_values[key_layer_idx + 1]
                key_state = self._bucket_slice_kv_cacheline(k_cache)
                value_state = self._bucket_slice_kv_cacheline(v_cache)

                past_key_values.append([key_state, value_state])

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(
            attention_mask, is_for_context_encoding, is_for_speculation, position_ids
        )
        active_mask = None
        if is_for_speculation:
            active_mask = torch.full(
                (self.speculation_length, self.speculation_length), True, device=attention_mask.device
            ).tril(diagonal=0)
            active_mask = active_mask[None, None, :, :].expand(
                self.batch_size, 1, self.speculation_length, self.speculation_length
            )

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
        )

        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
            k_cache = self._bucket_slice_kv_cacheline(self.past_key_values[idx*2])
            v_cache = self._bucket_slice_kv_cacheline(self.past_key_values[idx*2+1])

            if is_for_context_encoding:
                if self.config.is_continuous_batching:
                    # scatter back to the desired seq_ids
                    seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)
                    k_cache = torch.scatter(k_cache, 0, seq_id_index, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 0, seq_id_index, kv_per_layer[1])
                else:
                    # assign back to full kv_cacheline
                    k_cache = kv_per_layer[0]
                    v_cache = kv_per_layer[1]
            else:
                if self.padding_side == "left":
                    # TODO: fix it with scatter after right padding
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, kv_per_layer[0]], dim=2)
                    v_cache = torch.cat([v_cache, kv_per_layer[1]], dim=2)
                else:
                    scatter_index_new = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(
                        kv_per_layer[0]
                    )
                    k_cache = torch.scatter(k_cache, 2, scatter_index_new, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 2, scatter_index_new, kv_per_layer[1])

            k_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2, k_cache)
            v_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2 + 1, v_cache)

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            # speculative decoding case; only batch_size=1
            # will need to extend the logic to support multi-batch later
            # maybe just use position_ids for index?
            if position_ids.shape[-1] == self.speculation_length:
                index = torch.min(position_ids)
                index = torch.arange(index, index + self.speculation_length, device=hidden_states.device)
                index = index[None, :, None].expand(self.batch_size, self.speculation_length, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)
            else:
                # simple token generation
                index = torch.max(position_ids, dim=1, keepdim=True).indices
                index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
                hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        res = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            res = self.sampler.sample(logits[:, -1, :])

        return [res] + updated_kv_cache

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_model_output(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            active_mask: Optional[List[torch.FloatTensor]] = None,
    ):
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  #noqa
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
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

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
            )

            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        return (hidden_states, next_decoder_cache)


class NeuronBaseForCausalLM(NeuronSpeculation):
    _STATE_DICT_MODEL_PREFIX = "model."

    _model_cls = None

    def __init__(self, model_path: str, config: PretrainedConfig):
        super().__init__(config)

        self.config = config
        self.vocab_size = config.vocab_size
        self.padding_side = config.padding_side
        self.kv_cache_populated = False

        self.sampler = None

        self.models = []
        self.enable_context_encoding()
        if config.trace_tokengen_model:
            self.enable_token_generation()
        if config.speculation_length > 0:
            self.enable_speculation()
        if config.medusa_speculation_length > 0:
            self.enable_medusa_speculation()
        self.model_path = model_path

    @staticmethod
    def load_hf_model(model_path):
        raise NotImplementedError("load_hf_model is not implemented")

    def get_compiler_args(self):
        return None

    def enable_context_encoding(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.ctx_batch_size
        new_config.n_active_tokens = self.config.max_context_length
        new_config.bucket_n_active_tokens = True

        if not new_config.enable_bucketing:
            new_config.buckets = generate_buckets(new_config.max_context_length,new_config.max_context_length)
        else:
            new_config.buckets = generate_buckets(128, new_config.max_context_length)

        self.context_encoding_model = ModelWrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=CONTEXT_ENCODING_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.context_encoding_model)

    def enable_token_generation(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.tkg_batch_size
        new_config.n_active_tokens = 1
        new_config.bucket_n_active_tokens = False

        if not new_config.enable_bucketing:
            new_config.buckets = generate_buckets(new_config.max_length,new_config.max_length)
        else:
            new_config.buckets = generate_buckets(128, new_config.max_length)


        self.token_generation_model = ModelWrapper(
            config=new_config,
            model_cls=self._model_cls,
            tag=TOKEN_GENERATION_MODEL_TAG,
            compiler_args=self.get_compiler_args(),
        )
        self.models.append(self.token_generation_model)

    def enable_speculation(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.spec_batch_size
        new_config.n_active_tokens = self.config.speculation_length
        self.speculation_model = ModelWrapper(new_config, self._model_cls, tag=SPECULATION_MODEL_TAG)

        self.models.append(self.speculation_model)

    def enable_medusa_speculation(self):
        new_config = copy.deepcopy(self.config)
        new_config.batch_size = self.config.spec_batch_size
        new_config.n_active_tokens = self.config.medusa_speculation_length
        self.medusa_speculation_model = ModelWrapper(new_config, self._model_cls, tag=MEDUSA_MODEL_TAG)

        self.models.append(self.medusa_speculation_model)

    @classmethod
    def get_state_dict(cls, model_path: str, config: PretrainedConfig) -> dict:
        model_sd = load_state_dict(model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(cls._STATE_DICT_MODEL_PREFIX, "", 1)
                model_sd[updated_param_name] = model_sd[param_name]
                del model_sd[param_name]
        if os.path.exists(model_path + "/medusa_heads.pt"):
            medusa_head = torch.load(model_path + "/medusa_heads.pt", map_location="cpu")
            model_sd.update(medusa_head)
        return model_sd

    @classmethod
    def generate_quantized_state_dict(cls, model_path: str, config: PretrainedConfig) -> dict:
        hf_model = cls.load_hf_model(model_path)
        quantization_type = QuantizationType(config.quantization_type)
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_tensor_symmetric(float_model=hf_model, inplace=True)
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            hf_model_quant = quantize_pytorch_model_per_channel_symmetric(float_model=hf_model, inplace=True)
        else:
            raise RuntimeError(f"{config.quantization_type} not supported")

        model_quant_sd = hf_model_quant.model.state_dict()
        lm_head_quant_sd = hf_model_quant.lm_head.state_dict()
        convert_qint8_to_int8_state_dict(model_quant_sd)
        convert_qint8_to_int8_state_dict(lm_head_quant_sd)

        model_quant_sd["lm_head.weight"] = lm_head_quant_sd["weight"]
        model_quant_sd["lm_head.scale"] = lm_head_quant_sd["scale"]

        return model_quant_sd

    @classmethod
    def from_pretrained(cls, model_path: str, config: PretrainedConfig):
        return cls(model_path, config)

    def checkpoint_loader_fn(self, mmap: bool = False):
        # this function loads the model's state dictionary and weights from
        # the hf model
        if self.config.quantized is False:
            model_sd = self.get_state_dict(self.model_path, self.config)
            if self.config.torch_dtype == torch.bfloat16:
                for name, param in model_sd.items():
                    model_sd[name] = param.bfloat16()
            return model_sd
        return self.get_quantized_checkpoints()

    def get_quantized_checkpoints(self, mmap: bool = False):
        # this function loads the checkpointed float model state dictionary and weights
        # from the quantized hf model
        # This will be removed once we move to safe tensors in NxD
        existing_checkpoint_path = self.config.quantized_checkpoints_path
        if not os.path.exists(existing_checkpoint_path):
            raise FileNotFoundError(f"Quantized checkpoint file not found: {existing_checkpoint_path}")

        print(f"Using existing checkpoint: {existing_checkpoint_path}")
        model_quant_sd = torch.load(existing_checkpoint_path)

        # Make sure that the non quantized weights are in bfloat16 and not float32
        if self.config.torch_dtype == torch.bfloat16:
            for name, param in model_quant_sd.items():
                if param is not None and param.dtype == torch.float32:
                    if name.endswith(".scale"):
                        warnings.warn(f"Found float32 weights in quantized checkpoint: {name}. Will skip converting to bfloat16 as its scale")
                    else:
                        warnings.warn(f"Found float32 weights in quantized checkpoint: {name}. Will convert to bfloat16")
                        model_quant_sd[name] = param.bfloat16()

        return model_quant_sd

    def compile(self, serialize_base_path=None):

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        builder = ModelBuilder(
            router=None,
            tp_degree=self.config.tp_degree,
            checkpoint_loader=self.checkpoint_loader_fn,
            compiler_workdir=base_compile_work_dir
        )

        for model in self.models:
            builder.add(
                key=model.tag,
                model_instance=model.get_model_instance(),
                example_inputs=model.input_generator(),
                compiler_args=model.compiler_args,
                bucket_config=model.bucket_config,
                priority_model_idx=model.priority_model_idx,
            )

        traced_model = builder.trace(initialize_model_weights=False)
        torch.jit.save(traced_model, serialize_base_path + "model.pt")
        del traced_model

        builder.shard_checkpoint(serialize_path=os.path.join(serialize_base_path, "weights/"))
        self.is_loaded_to_neuron = True

    def load(self, serialize_base_path):

        traced_model = torch.jit.load(serialize_base_path + "model.pt")

        weights = []
        for rank in range(self.config.tp_degree):
            ckpt = load_file(os.path.join(serialize_base_path, f"weights/tp{rank}_sharded_checkpoint.safetensors"))
            weights.append(ckpt)

        traced_model.nxd_model.initialize(weights)

        for model_wrapper in self.models:
            model_wrapper.model = traced_model

    def to_neuron(self, serialize_base_path=None):
        if serialize_base_path is None:
            with tempfile.TemporaryDirectory(suffix="nxd-temp-serial-path") as tmpdirname:
                self.compile(tmpdirname)
                self.load(tmpdirname)
        else:
            self.compile(serialize_base_path)
            self.load(serialize_base_path)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            seq_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            medusa_args = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(output_attentions,
                                                                                       output_hidden_states,
                                                                                       return_dict)

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        self._log_input(input_ids, attention_mask, position_ids, seq_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        outputs, is_run_on_neuron = self._get_model_outputs(input_ids, attention_mask, position_ids, seq_ids, medusa_args)

        if self.config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            self._copy_past_key_values(outputs)

        if is_run_on_neuron:
            # When run on neuron, KV cache remains on device
            logits_or_next_tokens = outputs
        else:
            # When run on cpu, KV cache is returned which has to be ignored
            logits_or_next_tokens, *_ = outputs

        logging.debug("---output---")
        logging.debug(f"{'tokens' if self.config.on_device_sampling else 'logits'} = %s, ", logits_or_next_tokens)

        return self._construct_output(logits_or_next_tokens)

    def _setup_func_config(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return output_attentions, output_hidden_states, return_dict

    def _infer_attention_mask(self, position_ids):
        assert position_ids is not None, "need to call forward with position_ids if attention_mask is not provided"
        batch_size, seq_len = position_ids.shape
        if position_ids.shape[-1] == 1:
            seq_len = self.config.n_positions
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        else:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
        mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
        attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)
        return attention_mask

    def _log_input(self, input_ids, attention_mask, position_ids, seq_ids):
        logging.debug("---input---")
        logging.debug("input_ids shape = %s type=%s", input_ids.shape, input_ids.type())
        logging.debug("attention_mask shape = %s type=%s", attention_mask.shape, attention_mask.type())
        logging.debug("position_ids shape = %s type=%s", position_ids.shape, position_ids.type())
        logging.debug("input_ids =%s", input_ids)
        logging.debug("attention_mask =%s", attention_mask)
        logging.debug("position_ids =%s", position_ids)
        logging.debug(f"seq_ids: {seq_ids}")

        if self.config.trace_tokengen_model and not self.token_generation_model.is_neuron():
            logging.debug(f"first layer kv_cache: {self.token_generation_model.model.past_key_values[0][:, 0, :, 0]}")

    def _get_model_outputs(self, input_ids, attention_mask, position_ids, seq_ids, medusa_args):
        if (
            input_ids.shape[-1] > 1
            and input_ids.shape[-1] != self.config.speculation_length
            and input_ids.shape[-1] != self.config.medusa_speculation_length
        ):
            if self.config.is_medusa:
                medusa_args = self._prepare_inputs()
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                    *medusa_args,
                )
            else:
                outputs = self.context_encoding_model(
                    input_ids,
                    attention_mask,
                    position_ids,
                    seq_ids,
                )
            self.kv_cache_populated = True
            is_run_on_neuron = self.context_encoding_model.is_neuron()
        elif input_ids.shape[-1] == self.config.speculation_length:
            outputs = self.speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )
            is_run_on_neuron = self.speculation_model.is_neuron()
        elif input_ids.shape[-1] == self.config.medusa_speculation_length:
            outputs = self.medusa_speculation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
                *medusa_args,
            )
            is_run_on_neuron = self.medusa_speculation_model.is_neuron()
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )
            is_run_on_neuron = self.token_generation_model.is_neuron()

        return outputs, is_run_on_neuron

    def _copy_kv_cache(self, source_model, target_model):
        for source, target in zip(source_model.model.models, target_model.model.models):
            encoder_kv_cache_line = source.states
            token_gen_kv_cache_line = target.states
            for name, _ in token_gen_kv_cache_line._parameters.items():
                token_gen_kv_cache_line._parameters[name] = encoder_kv_cache_line._parameters[name]

    def _copy_past_key_values(self, outputs):
        new_past_key_values = outputs[1:]
        for i, new_past_key_value in enumerate(new_past_key_values):
            self.token_generation_model.model.past_key_values[i].data = new_past_key_value
            self.context_encoding_model.model.past_key_values[i].data = new_past_key_value

    def _construct_output(self, logits_or_next_tokens):
        if self.config.is_medusa:
            next_tokens = logits_or_next_tokens[:1, :, :]
        else:
            next_tokens = logits_or_next_tokens

        OutputParams = CausalLMOutputWithPast(
            logits=None if self.config.on_device_sampling else logits_or_next_tokens,
            hidden_states=logits_or_next_tokens,
            attentions=None,
        )

        if self.config.is_medusa:
            OutputParams.tokens = next_tokens[:1, :, :]
            OutputParams.medusa_tokens = next_tokens[1:, :, :]
        else:
            OutputParams.tokens = next_tokens

        return OutputParams

    # We override this function because we want to change the way attention_mask
    # is updated each iteration.
    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_for_token_generation: Optional[bool] = False,
            is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if is_for_token_generation:
                if self.padding_side == "left":
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                    attention_mask = attention_mask[:, 1:]
                else:
                    attention_mask = torch.cat(
                        [attention_mask.new_ones((attention_mask.shape[0], 1)), attention_mask], dim=-1
                    )
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if self.kv_cache_populated:
            input_ids = input_ids[:, -1:]

        accepted_indices = kwargs.get("accepted_indices", None)
        current_length = kwargs.get("current_length", None)
        medusa_mask = kwargs.get("medusa_mask", None)
        scatter_index = kwargs.get("scatter_index", None)

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if self.kv_cache_populated:
                position_ids = torch.amax(position_ids, 1, keepdim=True)
                position_ids = position_ids + 1

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", False),
                "attention_mask": attention_mask,
                "medusa_args": (accepted_indices, current_length, medusa_mask, scatter_index),
            }
        )
        return model_inputs

    def prepare_medusa_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if self.kv_cache_populated:
            input_ids = input_ids[:, -self.config.medusa_speculation_length :]
        position_ids = kwargs.get("position_ids")

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "medusa_args": (
                    kwargs.get("accepted_indices"),
                    kwargs.get("current_length"),
                    kwargs.get("medusa_mask"),
                    kwargs.get("scatter_index"),
                ),
            }
        )
        return model_inputs

    def _prepare_inputs(self):
        accepted_indices = torch.zeros((self.config.batch_size, self.config.num_medusa_heads + 1), dtype=torch.int64)
        current_length = torch.zeros((self.config.batch_size, self.config.num_medusa_heads + 1), dtype=torch.int64)
        medusa_mask = torch.zeros(
            (self.config.batch_size, self.config.medusa_speculation_length, self.config.medusa_speculation_length),
            dtype=torch.int64,
        )
        scatter_index = torch.zeros((self.config.batch_size, self.config.medusa_speculation_length), dtype=torch.int64)
        return accepted_indices, current_length, medusa_mask, scatter_index

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def reset(self):
        # We need to reset the KV cache flag for a new batch of inference.
        # When the flag is reset, the subsequent run will invoke the
        # context encoding model.
        self.kv_cache_populated = False

    def reset_kv_cache(self):
        # Zero out kv cache for debug.
        # For new batch inference, use reset() instead
        if not self.context_encoding_model.is_neuron():
            for i, kv_tensor in enumerate(self.context_encoding_model.model.past_key_values):
                self.context_encoding_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)

        if not self.token_generation_model.is_neuron():
            for i, kv_tensor in enumerate(self.token_generation_model.model.past_key_values):
                self.token_generation_model.model.past_key_values[i] = torch.zeros_like(kv_tensor)

    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_scores: Optional[bool] = None,
            output_logits: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        We override the GenerationMixin sample function (_sample for transformers>=4.39.0) to add support for right side padding.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        eos_token_id = 2
        pad_token_id = eos_token_id
        #pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        #eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False
        # auto-regressive generation
        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            is_for_token_generation = self.kv_cache_populated

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if not self.config.on_device_sampling:
                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)

                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)

            if not self.config.on_device_sampling:
                if self.sampler is None:
                    self.config.do_sample = True
                    self.sampler = Sampler(self.config)
                next_tokens = self.sampler.sample(outputs.logits[:, -1, :])
            else:
                next_tokens = outputs.tokens

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                is_for_token_generation=is_for_token_generation,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
            )
        else:
            return input_ids
