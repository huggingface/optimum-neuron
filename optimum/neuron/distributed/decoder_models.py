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

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from .base import Parallelizer
from .parallel_layers import ParallelCrossEntropy, ParallelEmbedding, ParallelMLP, ParallelSelfAttention
from .utils import create_sequence_parallel_attention_forward, linear_to_parallel_linear


if TYPE_CHECKING:
    import torch
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
    LAST_LINEAR_PROJECTION_NAME = "lm_head"


class GPTNeoParallelizer(Parallelizer):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "transformer.h.[0-9]+.ln_[1-2]",
        "transformer.ln_f",
    ]

    @classmethod
    def patch_attention_forward_for_sequence_parallelism(
        cls, model: "PreTrainedModel", sequence_parallel_enabled: bool
    ):
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention

        def _split_heads(self, tensor, num_heads, attn_head_size):
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            if sequence_parallel_enabled:
                # [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim] 
                return tensor.permute(1, 2, 0, 3).contiguous()
            return tensor.permute(0, 2, 1, 3)  

        def _merge_heads(self, tensor, num_heads, attn_head_size):
            if sequence_parallel_enabled:
                # [B, num_heads, S, head_dim] -> [S, B, num_heads, hidden_dim]
                tensor = tensor.permute(2, 0, 1, 3).contiguous()
            else:
                tensor = tensor.permute(0, 2, 1, 3).contiguous()
            new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
            return tensor.view(new_shape)

        for module in model.modules():
            if isinstance(module, GPTNeoSelfAttention):
                module._split_heads = _split_heads.__get__(module)
                module._merge_heads = _merge_heads.__get__(module)

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = GPTNeoParallelEmbedding.transform(model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
        for block in model.transformer.h:
            block.attn.attention = GPTNeoParallelSelfAttention.transform(
                model,
                block.attn.attention,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
            block.mlp = GPTNeoParallelMLP.transform(model, block.mlp, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
        if parallelize_embeddings:
            model = GPTNeoParallelCrossEntropy.transform(model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
        return model


class LlamaParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "model.embed_tokens"
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
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        # TODO: Make it smart by merging the gate and the up_proj.
        # WARNING: be careful of the interleaved outputs when doing TP!
        layer = super().transform(model, layer, sequence_parallel_enabled=sequence_parallel_enabled, device=device)

        weight_map = getattr(model, "_weight_map", None)

        module, attribute_name = cls._get_module_and_attribute_name(layer, "gate_proj")
        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        layer_qualified_name = ""
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
            linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
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
                device=device,
            ),
        )
        return layer


class LlamaParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = "lm_head"


class LlamaParallelizer(Parallelizer):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "model.layers.[0-9]+.input_layernorm",
        "model.layers.[0-9]+.post_attention_layernorm",
        "model.norm",
    ]

    @classmethod
    def patch_attention_forward_for_sequence_parallelism(
        cls, model: "PreTrainedModel", sequence_parallel_enabled: bool
    ):
        from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
        import torch.nn.functional as F
        from torch import nn
        import math

        def forward(
            self,
            hidden_states: "torch.Tensor",
            attention_mask: Optional["torch.Tensor"] = None,
            position_ids: Optional["torch.LongTensor"] = None,
            past_key_value: Optional[Tuple["torch.Tensor"]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
        ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], Optional[Tuple["torch.Tensor"]]]:
            if sequence_parallel_enabled:
                q_len, bsz, _ = hidden_states.size()
            else:
                bsz, q_len, _ = hidden_states.size()
            print(hidden_states.size())

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
                # [S, B, hidden_dim] -> [S, B, num_heads, head_dim] -> [B, num_heads, S, head_dim]
                query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(2, 0, 1, 3)
                key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(2, 0, 1, 3)
                value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(2, 0, 1, 3)
            else:
                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value[0].shape[-2]
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            if past_key_value is not None:
                # reuse k, v, self_attention
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states) if use_cache else None

            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            if sequence_parallel_enabled:
                # [B, S, hidden_dim] -> [S, B, hidden_dim]
                attn_output = attn_output.transpose(0, 1)

            return attn_output, attn_weights, past_key_value


        for module in model.modules():
            if isinstance(module, LlamaAttention):
                module.forward = forward.__get__(module)

    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = LlamaParallelEmbedding.transform(model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
        for layer in model.model.layers:
            layer.self_attn = LlamaParallelSelfAttention.transform(model, layer.self_attn, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
            layer.mlp = LLamaParallelMLP.transform(model, layer.mlp, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
        if parallelize_embeddings:
            model = LlamaParallelCrossEntropy.transform(model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device)
        return model
