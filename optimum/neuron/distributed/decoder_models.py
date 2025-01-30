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
"""Classes related to `neuronx-distributed` to perform parallelism."""

import math
from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock, GPTNeoSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForQuestionAnswering,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    repeat_kv,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralRMSNorm,
)

from .base import Parallelizer, PipelineParallelismSpecs, SequenceParallelismSpecs
from .parallel_layers import (
    LayerNormType,
    ParallelCrossEntropy,
    ParallelEmbedding,
    ParallelMLP,
    ParallelSelfAttention,
    ParallelSelfAttentionWithFusedQKV,
    SequenceCollectiveOpInfo,
)
from .utils import get_linear_weight_info, linear_to_parallel_linear


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class GPTNeoParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "transformer.wte"
    LM_HEAD_NAME = "lm_head"


class GPTNeoParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "out_proj"
    ALL_HEAD_SIZE_NAME = "embed_dim"


class GPTNeoParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "c_fc"
    SECOND_LINEAR_NAME = "c_proj"


class GPTNeoParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = {"GPTNeoForCausalLM": "lm_head"}


class GPTNeoSequenceParallelismSpecs(SequenceParallelismSpecs):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "transformer.h.[0-9]+.ln_[1-2]",
        "transformer.ln_f",
    ]
    SEQUENCE_COLLECTIVE_OPS_INFOS = [
        SequenceCollectiveOpInfo("scatter", GPTNeoBlock, "input", "first"),
        SequenceCollectiveOpInfo("gather", torch.nn.LayerNorm, "output", "last"),
    ]

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        if not sequence_parallel_enabled:
            return

        def _split_heads(self, tensor, num_heads, attn_head_size):
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            if sequence_parallel_enabled:
                # [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim]
                return tensor.permute(1, 2, 0, 3)
            return tensor.permute(0, 2, 1, 3)

        def _merge_heads(self, tensor, num_heads, attn_head_size):
            if sequence_parallel_enabled:
                # [B, num_heads, S, head_dim] -> [S, B, num_heads, head_dim]
                tensor = tensor.permute(2, 0, 1, 3).contiguous()
            else:
                tensor = tensor.permute(0, 2, 1, 3).contiguous()
            new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
            return tensor.view(new_shape)

        for module in model.modules():
            if isinstance(module, GPTNeoSelfAttention):
                module._split_heads = _split_heads.__get__(module)
                module._merge_heads = _merge_heads.__get__(module)


class GPTNeoParallelizer(Parallelizer):
    SEQUENCE_PARALLELSIM_SPECS_CLS = GPTNeoSequenceParallelismSpecs

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional[torch.device] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
        should_parallelize_layer_predicate_func: Optional[Callable[[torch.nn.Module], bool]] = None,
        **parallel_layer_specific_kwargs,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = GPTNeoParallelEmbedding.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        for block in model.transformer.h:
            block.attn.attention = GPTNeoParallelSelfAttention.transform(
                model,
                block.attn.attention,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
            block.mlp = GPTNeoParallelMLP.transform(
                model,
                block.mlp,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        if parallelize_embeddings:
            model = GPTNeoParallelCrossEntropy.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        return model


class GPTNeoXParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "gpt_neox.embed_in"
    LM_HEAD_NAME = "embed_out"


class GPTNeoXParallelSelfAttention(ParallelSelfAttentionWithFusedQKV):
    QUERY_KEY_VALUE_NAME = "query_key_value"
    OUTPUT_PROJECTION_NAME = "dense"
    NUM_ATTENTION_HEADS_NAME = "num_attention_heads"
    ALL_HEAD_SIZE_NAME = "hidden_size"


class GPTNeoXParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "dense_h_to_4h"
    SECOND_LINEAR_NAME = "dense_4h_to_h"


class GPTNeoXParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = {"GPTNeoXForCausalLM": "embed_out"}


class GPTNeoXSequenceParallelismSpecs(SequenceParallelismSpecs):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "gpt_neox.layers.[0-9]+.input_layernorm",
        "gpt_neox.layers.[0-9]+.post_attention_layernorm",
        "gpt_neox.final_layer_norm",
    ]
    SEQUENCE_COLLECTIVE_OPS_INFOS = [
        SequenceCollectiveOpInfo("scatter", "gpt_neox.embed_in", "output", "first"),
        SequenceCollectiveOpInfo("gather", torch.nn.LayerNorm, "output", "last"),
    ]

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        if not sequence_parallel_enabled:
            return

        from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

        def sequence_parallel_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: torch.FloatTensor,
            position_ids: torch.LongTensor,
            head_mask: Optional[torch.FloatTensor] = None,
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False,
        ):
            has_layer_past = layer_past is not None

            # Compute QKV
            # If sequence_parallel_enabled:
            #   --> [seq_len, batch, (num_heads * 3 * head_size)]
            # Else:
            #   --> [batch, seq_len, (num_heads * 3 * head_size)]
            qkv = self.query_key_value(hidden_states)

            # If sequence_parallel_enabled:
            #   --> [seq_len, batch, num_heads, 3 * head_size]
            # Else:
            #   --> [batch, seq_len, num_heads, 3 * head_size]
            new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
            qkv = qkv.view(*new_qkv_shape)

            if sequence_parallel_enabled:
                # [seq_len, batch, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
                query = qkv[..., : self.head_size].permute(1, 2, 0, 3)
                key = qkv[..., self.head_size : 2 * self.head_size].permute(1, 2, 0, 3)
                value = qkv[..., 2 * self.head_size :].permute(1, 2, 0, 3)
            else:
                # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
                query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
                key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
                value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

            # Compute rotary embeddings on rotary_ndims
            query_rot = query[..., : self.rotary_ndims]
            query_pass = query[..., self.rotary_ndims :]
            key_rot = key[..., : self.rotary_ndims]
            key_pass = key[..., self.rotary_ndims :]

            # Compute token offset for rotary embeddings (when decoding)
            seq_len = key.shape[-2]
            if has_layer_past:
                seq_len += layer_past[0].shape[-2]
            cos, sin = self.rotary_emb(value, seq_len=seq_len)
            query, key = apply_rotary_pos_emb(query_rot, key_rot, cos, sin, position_ids)
            query = torch.cat((query, query_pass), dim=-1)
            key = torch.cat((key, key_pass), dim=-1)

            # Cache QKV values
            if has_layer_past:
                past_key = layer_past[0]
                past_value = layer_past[1]
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
            present = (key, value) if use_cache else None

            # Compute attention
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

            # Reshape outputs
            if sequence_parallel_enabled:
                # [batch, num_attention_heads, seq_len, head_size] -> [seq_len, batch, hidden_size]
                attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
                attn_output = attn_output.view(*attn_output.shape[:2], -1)
            else:
                attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
            attn_output = self.dense(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)

            return outputs

        for module in model.modules():
            if isinstance(module, GPTNeoXAttention):
                module.forward = sequence_parallel_forward.__get__(module)


class GPTNeoXParallelizer(Parallelizer):
    SEQUENCE_PARALLELSIM_SPECS_CLS = GPTNeoXSequenceParallelismSpecs

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional[torch.device] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
        should_parallelize_layer_predicate_func: Optional[Callable[[torch.nn.Module], bool]] = None,
        **parallel_layer_specific_kwargs,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = GPTNeoXParallelEmbedding.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        for layer in model.gpt_neox.layers:
            layer.attention = GPTNeoXParallelSelfAttention.transform(
                model,
                layer.attention,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
            layer.mlp = GPTNeoXParallelMLP.transform(
                model,
                layer.mlp,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        if parallelize_embeddings:
            model = GPTNeoXParallelCrossEntropy.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        return model


class LlamaParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = {
        "default": "model.embed_tokens",
        "LlamaForQuestionAnswering": "transformer.embed_tokens",
    }
    LM_HEAD_NAME = "lm_head"


class LlamaParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "o_proj"
    NUM_ATTENTION_HEADS_NAME = "num_heads"
    NUM_KEY_VALUE_HEADS_NAME = "num_key_value_heads"
    NUM_KEY_VALUE_GROUPS_NAME = "num_key_value_groups"
    ALL_HEAD_SIZE_NAME = "hidden_size"


class LLamaParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "up_proj"
    SECOND_LINEAR_NAME = "down_proj"

    @classmethod
    def _transform(
        cls,
        model: "PreTrainedModel",
        layer: torch.nn.Module,
        sequence_parallel_enabled: bool = False,
        device: Optional[torch.device] = None,
        **parallel_layer_specific_kwargs,
    ) -> torch.nn.Module:
        # TODO: Make it smart by merging the gate and the up_proj.
        # WARNING: be careful of the interleaved outputs when doing TP!
        layer = super()._transform(
            model,
            layer,
            sequence_parallel_enabled=sequence_parallel_enabled,
            device=device,
            **parallel_layer_specific_kwargs,
        )

        skip_linear_weight_load = parallel_layer_specific_kwargs["skip_linear_weight_load"]

        weight_map = getattr(model, "_weight_map", None)

        module, attribute_name = cls._get_module_and_attribute_name(layer, "gate_proj")
        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        layer_qualified_name = ""
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
            linear_layer_weight_info, linear_layer_bias_weight_info = get_linear_weight_info(
                weight_map,
                f"{layer_qualified_name}.{attribute_name}",
                device=device,
            )

        setattr(
            module,
            attribute_name,
            linear_to_parallel_linear(
                getattr(module, attribute_name),
                "column",
                gather_output=False,
                linear_layer_weight_info=linear_layer_weight_info,
                linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                sequence_parallel_enabled=sequence_parallel_enabled,
                skip_weight_load=skip_linear_weight_load,
                device=device,
            ),
        )
        return layer


class LlamaParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = {
        "LlamaForCausalLM": "lm_head",
    }


class LlamaSequenceParallelismSpecs(SequenceParallelismSpecs):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "(model|transformer).layers.[0-9]+.input_layernorm",
        "(model|transformer).layers.[0-9]+.post_attention_layernorm",
        "(model|transformer).norm",
    ]
    LAYERNORM_TYPE = LayerNormType.RMS_NORM
    SEQUENCE_COLLECTIVE_OPS_INFOS = [
        SequenceCollectiveOpInfo("scatter", LlamaDecoderLayer, "input", "first"),
        SequenceCollectiveOpInfo("gather", LlamaRMSNorm, "output", "last"),
    ]

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        if not sequence_parallel_enabled:
            return

        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        def attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = False,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            if self.config.pretraining_tp > 1:
                key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                query_slices = self.q_proj.weight.split(
                    (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                )
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)

            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            if sequence_parallel_enabled:
                q_len, bsz, _ = query_states.size()
            else:
                bsz, q_len, _ = query_states.size()

            if sequence_parallel_enabled:
                # [S, B, hidden_dim] -> [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim]
                query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
                key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
                value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(
                    1, 2, 0, 3
                )
            else:
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            past_key_value = getattr(self, "past_key_value", past_key_value)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            if sequence_parallel_enabled:
                # [B, num_heads, S, head_dim] -> [S, B, num_heads, head_dim]
                attn_output = attn_output.permute(2, 0, 1, 3)
                attn_output = attn_output.reshape(q_len, bsz, -1)
            else:
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum(
                    [F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
                )
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights

        for module in model.modules():
            if isinstance(module, LlamaAttention):
                module.forward = attention_forward.__get__(module)


class LlamaPipelineParallelismSpecs(PipelineParallelismSpecs):
    TRASNFORMER_LAYER_CLS = LlamaDecoderLayer
    DEFAULT_INPUT_NAMES = {
        "default": ("input_ids", "attention_mask", "labels"),
        "LlamaForQuestionAnswering": ("input_ids", "attention_mask", "start_positions", "end_positions"),
    }

    LEAF_MODULE_CLASSES_NAMES = [LlamaRMSNorm, LlamaRotaryEmbedding]


class LlamaParallelizer(Parallelizer):
    SEQUENCE_PARALLELSIM_SPECS_CLS = LlamaSequenceParallelismSpecs
    PIPELINE_PARALLELISM_SPECS_CLS = LlamaPipelineParallelismSpecs

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional[torch.device] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
        should_parallelize_layer_predicate_func: Optional[Callable[[torch.nn.Module], bool]] = None,
        **parallel_layer_specific_kwargs,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = LlamaParallelEmbedding.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )

        # The name of the LlamaModel attribute depends on the task.
        # It is "model" for every task except question-answering where it is "transformer".
        if isinstance(model, LlamaForQuestionAnswering):
            layers = model.transformer.layers
        else:
            layers = model.model.layers

        for layer in layers:
            # FIXME: temporary workaround to avoid too many changes in the transformation code
            layer.self_attn.num_heads = layer.self_attn.config.num_attention_heads
            layer.self_attn.num_key_value_heads = layer.self_attn.config.num_key_value_heads
            layer.self_attn.hidden_size = layer.self_attn.config.hidden_size
            layer.self_attn = LlamaParallelSelfAttention.transform(
                model,
                layer.self_attn,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
            layer.mlp = LLamaParallelMLP.transform(
                model,
                layer.mlp,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        if parallelize_embeddings:
            LlamaParallelEmbedding.overwrite_vocab_size_value_for_cross_entropy_computation(model)
            model = LlamaParallelCrossEntropy.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        return model


class MistralParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "model.embed_tokens"
    LM_HEAD_NAME = "lm_head"


class MistralParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "o_proj"
    NUM_ATTENTION_HEADS_NAME = "num_heads"
    NUM_KEY_VALUE_HEADS_NAME = "num_key_value_heads"
    NUM_KEY_VALUE_GROUPS_NAME = "num_key_value_groups"
    ALL_HEAD_SIZE_NAME = "hidden_size"


class MistralParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "up_proj"
    SECOND_LINEAR_NAME = "down_proj"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: torch.nn.Module,
        sequence_parallel_enabled: bool = False,
        device: Optional[torch.device] = None,
        should_parallelize_layer_predicate_func: Optional[Callable[[torch.nn.Module], bool]] = None,
        **parallel_layer_specific_kwargs,
    ) -> torch.nn.Module:
        if should_parallelize_layer_predicate_func is not None and not should_parallelize_layer_predicate_func(layer):
            return layer
        # TODO: Make it smart by merging the gate and the up_proj.
        # WARNING: be careful of the interleaved outputs when doing TP!
        layer = super().transform(
            model,
            layer,
            sequence_parallel_enabled=sequence_parallel_enabled,
            device=device,
            **parallel_layer_specific_kwargs,
        )

        skip_linear_weight_load = parallel_layer_specific_kwargs["skip_linear_weight_load"]
        weight_map = getattr(model, "_weight_map", None)

        module, attribute_name = cls._get_module_and_attribute_name(layer, "gate_proj")
        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        layer_qualified_name = ""
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
            linear_layer_weight_info, linear_layer_bias_weight_info = get_linear_weight_info(
                weight_map,
                f"{layer_qualified_name}.{attribute_name}",
                device=device,
            )

        setattr(
            module,
            attribute_name,
            linear_to_parallel_linear(
                getattr(module, attribute_name),
                "column",
                gather_output=False,
                linear_layer_weight_info=linear_layer_weight_info,
                linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                sequence_parallel_enabled=sequence_parallel_enabled,
                skip_weight_load=skip_linear_weight_load,
                device=device,
            ),
        )
        return layer


class MistralParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = {"MistralForCausalLM": "lm_head"}


class MistralSequenceParallelismSpecs(SequenceParallelismSpecs):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "model.layers.[0-9]+.input_layernorm",
        "model.layers.[0-9]+.post_attention_layernorm",
        "model.norm",
    ]
    LAYERNORM_TYPE = LayerNormType.RMS_NORM
    SEQUENCE_COLLECTIVE_OPS_INFOS = [
        SequenceCollectiveOpInfo("scatter", MistralDecoderLayer, "input", "first"),
        SequenceCollectiveOpInfo("gather", MistralRMSNorm, "output", "last"),
    ]

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        if not sequence_parallel_enabled:
            return

        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb

        def attention_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            if sequence_parallel_enabled:
                q_len, bsz, _ = query_states.size()
            else:
                bsz, q_len, _ = query_states.size()

            if sequence_parallel_enabled:
                # [S, B, hidden_dim] -> [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim]
                query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
                key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
                value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(
                    1, 2, 0, 3
                )
            else:
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            if sequence_parallel_enabled:
                # [B, num_heads, S, head_dim] -> [S, B, num_heads, head_dim]
                attn_output = attn_output.permute(2, 0, 1, 3)
                attn_output = attn_output.reshape(q_len, bsz, -1)
            else:
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights

        for module in model.modules():
            if isinstance(module, MistralAttention):
                module.forward = attention_forward.__get__(module)


class MistralParallelizer(Parallelizer):
    SEQUENCE_PARALLELSIM_SPECS_CLS = MistralSequenceParallelismSpecs

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional[torch.device] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
        should_parallelize_layer_predicate_func: Optional[Callable[[torch.nn.Module], bool]] = None,
        **parallel_layer_specific_kwargs,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = MistralParallelEmbedding.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        for layer in model.model.layers:
            # FIXME: temporary workaround to avoid too many changes in the transformation code
            layer.self_attn.num_heads = layer.self_attn.config.num_attention_heads
            layer.self_attn.num_key_value_heads = layer.self_attn.config.num_key_value_heads
            layer.self_attn.hidden_size = layer.self_attn.config.hidden_size
            layer.self_attn = MistralParallelSelfAttention.transform(
                model,
                layer.self_attn,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
            layer.mlp = MistralParallelMLP.transform(
                model,
                layer.mlp,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        if parallelize_embeddings:
            MistralParallelEmbedding.overwrite_vocab_size_value_for_cross_entropy_computation(model)
            model = MistralParallelCrossEntropy.transform(
                model,
                model,
                sequence_parallel_enabled=sequence_parallel_enabled,
                should_parallelize_layer_predicate_func=should_parallelize_layer_predicate_func,
                device=device,
                **parallel_layer_specific_kwargs,
            )
        return model
