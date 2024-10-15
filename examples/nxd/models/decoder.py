from typing import List, Optional, Union

import torch
from models.gqa import (  # noqa: E402
    determine_sharding_strategy,  # noqa: E402
    get_shardable_head_counts,  # noqa: E402
)  # noqa: E402
from modules.autobucketing import slice_lhs, slice_rhs  # noqa: E402
from modules.sampling import Sampler  # noqa: E402
from neuronx_distributed.parallel_layers import parallel_state, utils  # noqa: E402
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import (
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
)


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


class NeuronDecoderModel(PreTrainedModel):
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

    def _create_simple_attn_mask(self, attention_mask):
        return attention_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.n_positions).to(torch.bool)

    def create_attn_mask(self, attention_mask, is_for_context_encoding, position_ids):
        if is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask)
        else:
            return self._create_simple_attn_mask(attention_mask)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        accepted_indices=None,
        current_length=None,
        scatter_index=None,
    ):

        is_for_context_encoding = input_ids.shape[-1] > 1

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
        attention_mask = self.create_attn_mask(attention_mask, is_for_context_encoding, position_ids)
        active_mask = None

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
        )

        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
            k_cache = self._bucket_slice_kv_cacheline(self.past_key_values[idx * 2])
            v_cache = self._bucket_slice_kv_cacheline(self.past_key_values[idx * 2 + 1])

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
                    scatter_index_new = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(kv_per_layer[0])
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
            device = input_ids.device if input_ids is not None else inputs_embeds.device  # noqa
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
