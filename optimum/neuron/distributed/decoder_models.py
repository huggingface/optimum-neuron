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

from typing import TYPE_CHECKING, Callable, Optional

import torch
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock, GPTNeoSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForQuestionAnswering,
    LlamaRMSNorm,
)
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralDecoderLayer,
    MistralRMSNorm,
)

from ..models.core import NeuronAttention
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

        for module in model.modules():
            if isinstance(module, GPTNeoXAttention):
                if not isinstance(module, NeuronAttention):
                    raise ValueError(
                        "The gpt neox model has not been prepared by the NeuronPreparator. It is required for sequence "
                        "parallelism."
                    )
                module.sequence_parallel_enabled = sequence_parallel_enabled


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
        for module in model.modules():
            if isinstance(module, LlamaAttention):
                if not isinstance(module, NeuronAttention):
                    raise ValueError(
                        "The llama model has not been prepared by the NeuronPreparator. It is required for sequence "
                        "parallelism."
                    )
                module.sequence_parallel_enabled = sequence_parallel_enabled


class LlamaPipelineParallelismSpecs(PipelineParallelismSpecs):
    TRASNFORMER_LAYER_CLS = LlamaDecoderLayer
    DEFAULT_INPUT_NAMES = {
        "default": ("input_ids", "attention_mask", "labels"),
        "LlamaForQuestionAnswering": ("input_ids", "attention_mask", "start_positions", "end_positions"),
    }

    LEAF_MODULE_CLASSES_NAMES = [LlamaRMSNorm]


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

        for module in model.modules():
            if isinstance(module, MistralAttention):
                if not isinstance(module, NeuronAttention):
                    raise ValueError(
                        "The mistral model has not been prepared by the NeuronPreparator. It is required for sequence "
                        "parallelism."
                    )
                module.sequence_parallel_enabled = sequence_parallel_enabled


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
