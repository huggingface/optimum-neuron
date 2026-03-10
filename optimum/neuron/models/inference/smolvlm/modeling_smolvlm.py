# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch SmolVLM (Idefics3) model for NXD inference."""

import gc
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Idefics3ForConditionalGeneration, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector

from ..backend.config import NxDVLMNeuronConfig
from ..backend.modules.decoder import NxDModelForCausalLM
from ..backend.modules.decoder.decoder_wrappers import (
    CHUNKED_PREFILL_MODEL_TAG,
    CONTEXT_ENCODING_MODEL_TAG,
    TOKEN_GENERATION_MODEL_TAG,
)
from ..backend.modules.decoder.vlm_builders import (
    NxDVLMChunkedPrefillBuilder,
    NxDVLMContextEncodingBuilder,
    NxDVLMTokenGenerationBuilder,
)
from ..backend.modules.decoder.vlm_wrappers import NxDVLMContextDecoderWrapper, NxDVLMTokenGenerationWrapper
from ..backend.modules.generation.sampling import validate_sampling_params
from ..backend.pretrained_model import NxDPreTrainedModel, normalize_path
from ..llama.modeling_llama import NxDLlamaModel, convert_state_dict_to_fused_qkv


logger = logging.getLogger("Neuron")

VISION_ENCODER_FILE_NAME = "vision_encoder.pt"


class NeuronSigLIPVisionEmbeddings(nn.Module):
    """Idefics3VisionEmbeddings for fixed compiled image size.

    Uses the same attribute names (patch_embedding, position_embedding) as
    Idefics3VisionEmbeddings so HF state-dict keys map directly on load.

    Position IDs are pre-computed using the same fractional-coordinate bucketing
    as HF's Idefics3VisionEmbeddings (for a full patch_attention_mask). This
    ensures the position embeddings match the HF reference exactly.
    """

    def __init__(self, vision_config):
        super().__init__()
        num_channels = getattr(vision_config, "num_channels", 3)
        embed_dim = vision_config.hidden_size
        patch_size = vision_config.patch_size
        image_size = vision_config.image_size
        num_patches_per_side = image_size // patch_size
        num_patches = num_patches_per_side**2

        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding="valid",
        )
        self.position_embedding = nn.Embedding(num_patches, embed_dim)

        # Pre-compute position IDs matching HF's fractional bucketing scheme.
        boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
        h_indices = torch.arange(num_patches_per_side, dtype=torch.float32)
        w_indices = torch.arange(num_patches_per_side, dtype=torch.float32)
        fractional_h = h_indices / num_patches_per_side * (1 - 1e-6)
        fractional_w = w_indices / num_patches_per_side * (1 - 1e-6)
        bucket_h = torch.bucketize(fractional_h, boundaries, right=True)
        bucket_w = torch.bucketize(fractional_w, boundaries, right=True)
        position_ids = (bucket_h[:, None] * num_patches_per_side + bucket_w).flatten()
        self.register_buffer("position_ids", position_ids.unsqueeze(0), persistent=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        hidden_states = patch_embeds.flatten(2).transpose(1, 2)
        return hidden_states + self.position_embedding(self.position_ids.expand(batch_size, -1))


class NeuronSigLIPMLP(nn.Module):
    """MLP block for the vision encoder."""

    def __init__(self, vision_config):
        super().__init__()

        hidden_size = vision_config.hidden_size
        intermediate_size = vision_config.intermediate_size
        self.activation_fn = ACT2FN[vision_config.hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class NeuronSigLIPAttention(nn.Module):
    """Multi-headed attention for the vision encoder."""

    def __init__(self, vision_config):
        super().__init__()

        self.embed_dim = vision_config.hidden_size
        self.num_heads = vision_config.num_attention_heads
        self.head_dim = vision_config.hidden_size // vision_config.num_attention_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Reshape into (batch, heads, seq, head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(queries.dtype)
        attn_output = torch.matmul(attn_weights, values)

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.num_heads * self.head_dim)
        return self.out_proj(attn_output)


class NeuronSigLIPEncoderLayer(nn.Module):
    """Single Idefics3 vision encoder layer."""

    def __init__(self, vision_config):
        super().__init__()
        embed_dim = vision_config.hidden_size
        self.self_attn = NeuronSigLIPAttention(vision_config)
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=vision_config.layer_norm_eps)
        self.mlp = NeuronSigLIPMLP(vision_config)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=vision_config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class NeuronSigLIPEncoder(nn.Module):
    """Stack of NeuronSigLIPEncoderLayer."""

    def __init__(self, vision_config):
        super().__init__()
        self.layers = nn.ModuleList(
            [NeuronSigLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class NeuronSigLIPVisionTransformer(nn.Module):
    """Idefics3 vision transformer.

    Embeddings (Conv2d patch embedding + position embedding) and all encoder
    linear layers use standard nn.Linear (no TP-sharded modules). The vision
    encoder is compiled with the configured tensor parallel degree.
    """

    def __init__(self, vision_config):
        super().__init__()
        embed_dim = vision_config.hidden_size
        self.embeddings = NeuronSigLIPVisionEmbeddings(vision_config)
        self.encoder = NeuronSigLIPEncoder(vision_config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=vision_config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        return self.post_layernorm(hidden_states)


class NeuronIdefics3VisionEncoder(nn.Module):
    """Idefics3 vision encoder + connector.

    All linear layers use standard nn.Linear (no TP-sharded modules). The
    connector is the HF ``Idefics3Connector`` implementation.
    """

    def __init__(self, config):
        super().__init__()
        vision_config = config.vision_config

        self.vision_model = NeuronSigLIPVisionTransformer(vision_config)
        self.connector = Idefics3Connector(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.vision_model(pixel_values)
        return self.connector(hidden)

    @classmethod
    def make_checkpoint_loader(cls, model_id: str, token=None):
        """Return a callable that loads HF weights into this module's key namespace."""

        def loader():
            logger.info(f"Loading vision encoder weights from {model_id}")
            hf = Idefics3ForConditionalGeneration.from_pretrained(model_id, token=token, low_cpu_mem_usage=True)
            sd = {}
            # hf.model.vision_model is Idefics3VisionTransformer; its state_dict keys are
            # embeddings.*, encoder.layers.{i}.*, post_layernorm.* — matching our NeuronSigLIPVisionTransformer
            for k, v in hf.model.vision_model.state_dict().items():
                sd[f"vision_model.{k}"] = v
            # hf.model.connector is Idefics3Connector; key: modality_projection.proj.weight
            for k, v in hf.model.connector.state_dict().items():
                sd[f"connector.{k}"] = v
            del hf
            gc.collect()
            return sd

        return loader


class NxDSmolVLMDecoderModel(NxDLlamaModel):
    """Text decoder for SmolVLM (Idefics3).

    Receives the full ``Idefics3Config`` and extracts ``text_config`` before
    delegating to the standard Llama-style layer construction.
    """

    def __init__(self, config: PretrainedConfig, neuron_config: NxDVLMNeuronConfig):
        text_config = getattr(config, "text_config", config)
        super().__init__(text_config, neuron_config)


class SmolVLMNxDModelForCausalLM(NxDModelForCausalLM):
    """NxD model for SmolVLM (Idefics3) vision-language inference.

    Manages two compiled artifacts:

        * ``model.pt`` — decoder graph bundle for context encoding / token generation
            (speculation path depends on runtime/base configuration).
        * ``vision_encoder.pt`` — compiled SigLIP vision encoder + HF
            ``Idefics3Connector``.

    At context/chunked-prefill time the decoder computes text embeddings on
    Neuron, then injects vision features at ``image_token_id`` positions.
    Token generation uses the standard ``input_ids`` path.
    """

    _model_cls = NxDSmolVLMDecoderModel
    _STATE_DICT_MODEL_PREFIX = "model.text_model."
    task = "image-text-to-text"

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDVLMNeuronConfig,
        traced_model: torch.jit.ScriptModule,
        graph_builders,
        traced_vision_encoder: torch.jit.ScriptModule = None,
    ):
        super().__init__(config, neuron_config, traced_model, graph_builders)
        self._traced_vision_encoder = traced_vision_encoder

        # Replace the default context_encoding_model with the VLM-aware wrapper
        ctx_neuron_config = NxDModelForCausalLM._create_context_encoding_config(neuron_config)
        self.context_encoding_model = NxDVLMContextDecoderWrapper(
            config=config,
            neuron_config=ctx_neuron_config,
            model=traced_model,
            tag=CONTEXT_ENCODING_MODEL_TAG,
        )

        if neuron_config.prefill_chunk_size > 0:
            chunk_neuron_config = NxDModelForCausalLM._create_chunked_prefill_config(neuron_config)
            self.chunked_prefill_model = NxDVLMContextDecoderWrapper(
                config=config,
                neuron_config=chunk_neuron_config,
                model=traced_model,
                tag=CHUNKED_PREFILL_MODEL_TAG,
            )

        # Replace the default token_generation_model with the VLM-aware wrapper that passes
        # dummy image-injection tensors to match the compiled signature.
        tkg_neuron_config = NxDModelForCausalLM._create_token_generation_config(neuron_config)
        self.token_generation_model = NxDVLMTokenGenerationWrapper(
            config=config,
            neuron_config=tkg_neuron_config,
            model=traced_model,
            tag=TOKEN_GENERATION_MODEL_TAG,
        )

        self.image_token_id = getattr(config, "image_token_id", None)

    # ------------------------------------------------------------------
    # Graph builder / compilation
    # ------------------------------------------------------------------

    @classmethod
    def create_graph_builders(cls, config, neuron_config):
        tkg_neuron_config = NxDModelForCausalLM._create_token_generation_config(neuron_config)
        graph_builders = {}
        if neuron_config.prefill_chunk_size > 0:
            chunk_neuron_config = NxDModelForCausalLM._create_chunked_prefill_config(neuron_config)
            graph_builders["chunked_prefill"] = NxDVLMChunkedPrefillBuilder(
                config=config,
                neuron_config=chunk_neuron_config,
                max_tokens=chunk_neuron_config.sequence_length,
                active_tokens=chunk_neuron_config.prefill_chunk_size,
                model_cls=cls._model_cls,
            )
        else:
            ctx_neuron_config = NxDModelForCausalLM._create_context_encoding_config(neuron_config)
            graph_builders["context_encoding"] = NxDVLMContextEncodingBuilder(
                config=config,
                neuron_config=ctx_neuron_config,
                max_tokens=ctx_neuron_config.max_context_length,
                active_tokens=ctx_neuron_config.max_context_length,
                model_cls=cls._model_cls,
            )

        graph_builders["token_generation"] = NxDVLMTokenGenerationBuilder(
            config=config,
            neuron_config=tkg_neuron_config,
            max_tokens=tkg_neuron_config.sequence_length,
            active_tokens=1,
            model_cls=cls._model_cls,
            priority_model_idx=0,
        )
        return graph_builders

    @classmethod
    def _compile_vision_encoder(cls, model_id, config, neuron_config, token=None):
        """Compile the Idefics3 vision encoder + connector as a single Neuron model.

        Uses ModelBuilder and compiles the vision subgraph before decoder export
        to avoid XLA/process-state conflicts during end-to-end export.

        Weights are always loaded and saved via initialize_with_saved_weights so that
        torch.jit.save embeds them in vision_encoder.pt.  On load, _from_pretrained
        calls initialize_with_saved_weights again to restore them to the Neuron device.
        """
        from neuronx_distributed.trace.model_builder import BaseModelInstance, ModelBuilder

        tp_degree = neuron_config.tp_degree
        dtype = neuron_config.torch_dtype

        def module_cls():
            m = NeuronIdefics3VisionEncoder(config).eval()
            if dtype != torch.float32:
                m = m.to(dtype)
            return m

        model_instance = BaseModelInstance(module_cls=module_cls, input_output_aliases={})

        batch_size = neuron_config.batch_size * neuron_config.max_num_images
        example_input = torch.zeros(
            (batch_size, 3, neuron_config.image_size, neuron_config.image_size),
            dtype=dtype,
        )

        def checkpoint_loader():
            from huggingface_hub import snapshot_download

            from ..backend.modules.checkpoint import load_state_dict

            # Resolve to a local directory — snapshot_download returns a cached local path.
            if os.path.isdir(model_id):
                local_path = model_id
            else:
                local_path = snapshot_download(
                    repo_id=model_id,
                    token=token,
                    allow_patterns=["*.safetensors*", "*.bin*", "*.json"],
                )

            # Load the full raw state-dict from disk (safetensors/bin files).
            # This avoids from_pretrained which allocates via torch.empty and is
            # affected by the init_on_device("meta") context used by shard_checkpoint.
            full_sd = load_state_dict(local_path)

            sd = {}
            for k, v in full_sd.items():
                if k.startswith("model.vision_model."):
                    sd["vision_model." + k[len("model.vision_model.") :]] = v
                elif k.startswith("model.connector."):
                    sd["connector." + k[len("model.connector.") :]] = v
            return sd

        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_vision_encoder/")
        builder = ModelBuilder(
            router=None,
            tp_degree=tp_degree,
            checkpoint_loader=checkpoint_loader,
            pp_degree=1,
            ep_degree=1,
            local_ranks_size=tp_degree,
            world_size=tp_degree,
            compiler_workdir=base_compile_work_dir,
            logical_nc_config=getattr(neuron_config, "logical_nc_config", 1),
            weights_to_skip_layout_optimization=getattr(neuron_config, "weights_to_skip_layout_optimization", set()),
        )
        builder.add(
            key="vision_encoder",
            model_instance=model_instance,
            example_inputs=[(example_input,)],
            compiler_args=cls.get_compiler_args(neuron_config),
            priority_model_idx=0,
        )

        logger.info("Compiling vision encoder with ModelBuilder (including weights) …")
        traced = builder.trace(initialize_model_weights=True)
        gc.collect()
        return traced

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def generate(self, input_ids, attention_mask=None, pixel_values=None, generation_config=None, **kwargs):
        """Override generate to capture pixel_values for the context encoding forward pass.

        The base ``_sample()`` method does not pass ``pixel_values`` through to ``forward()``.
        We store them temporarily so ``forward()`` can read them during context encoding.
        """
        self._current_pixel_values = pixel_values
        try:
            return super().generate(
                input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs
            )
        finally:
            self._current_pixel_values = None

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        seq_ids: torch.LongTensor = None,
        sampling_params: torch.FloatTensor = None,
        pixel_values: torch.FloatTensor = None,
        **kwargs,
    ):
        # Retrieve pixel_values stored by generate() if not passed directly
        if pixel_values is None:
            pixel_values = getattr(self, "_current_pixel_values", None)

        if self.neuron_config.on_device_sampling:
            validate_sampling_params(sampling_params, self.neuron_config.max_topk)

        is_context_encoding = input_ids.shape[-1] > 1 and not position_ids.min().item()
        is_speculation = input_ids.shape[-1] == self.neuron_config.speculation_length

        if is_context_encoding:
            if self.neuron_config.prefill_chunk_size > 0:
                chunk_size = self.neuron_config.prefill_chunk_size
                outputs = None
                for chunk_start in range(0, input_ids.shape[-1], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, input_ids.shape[-1])
                    chunk_ids = input_ids[:, chunk_start:chunk_end]
                    chunk_pos = position_ids[:, chunk_start:chunk_end]
                    chunk_image_embeds, chunk_image_token_mask = self._prepare_image_injection_tensors(
                        chunk_ids, pixel_values
                    )
                    outputs = self.chunked_prefill_model(
                        chunk_ids,
                        chunk_pos,
                        seq_ids,
                        sampling_params,
                        chunk_image_embeds,
                        chunk_image_token_mask,
                    )
            else:
                image_embeds, image_token_mask = self._prepare_image_injection_tensors(input_ids, pixel_values)
                outputs = self.context_encoding_model(
                    input_ids,
                    position_ids,
                    seq_ids,
                    sampling_params,
                    image_embeds,
                    image_token_mask,
                )
            self.kv_cache_populated = True
        elif is_speculation:
            outputs = self.speculation_model(input_ids, position_ids, seq_ids, sampling_params)
        else:
            outputs = self.token_generation_model(input_ids, position_ids, seq_ids, sampling_params)

        return outputs

    def prefill_chunk(self, input_ids, position_ids, seq_ids, sampling_params):
        """Process one prompt chunk while injecting image features on-device."""
        pixel_values = getattr(self, "_current_pixel_values", None)
        image_embeds, image_token_mask = self._prepare_image_injection_tensors(input_ids, pixel_values)
        outputs = self.chunked_prefill_model(
            input_ids,
            position_ids,
            seq_ids,
            sampling_params,
            image_embeds,
            image_token_mask,
        )
        self.kv_cache_populated = True
        return outputs

    def _prepare_image_injection_tensors(
        self, input_ids: torch.Tensor, pixel_values: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build image injection tensors for decoder-side on-device embedding.

        Returns:
            image_embeds: ``[B, S, H]`` tensor with image features at token positions.
            image_token_mask: ``[B, S]`` bool mask selecting those positions.
        """
        text_config = getattr(self.config, "text_config", self.config)
        image_embeds = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], text_config.hidden_size),
            dtype=self.neuron_config.torch_dtype,
            device=input_ids.device,
        )
        image_token_mask = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1]),
            dtype=torch.bool,
            device=input_ids.device,
        )

        if pixel_values is not None and self._traced_vision_encoder is not None and self.image_token_id is not None:
            # pixel_values: [B, N, C, H, W] or [B, C, H, W]
            if pixel_values.dim() == 5:
                pixel_values_flat = pixel_values.reshape(-1, *pixel_values.shape[2:])
            else:
                pixel_values_flat = pixel_values
            pixel_values_flat = pixel_values_flat.to(self.neuron_config.torch_dtype)

            # Pad to the compiled vision encoder batch size (batch_size * max_num_images).
            # The processor may produce fewer tiles than the compiled max.
            compiled_batch = self.neuron_config.batch_size * self.neuron_config.max_num_images
            num_real_tiles = pixel_values_flat.shape[0]
            if num_real_tiles < compiled_batch:
                padding = torch.zeros(
                    compiled_batch - num_real_tiles,
                    *pixel_values_flat.shape[1:],
                    dtype=pixel_values_flat.dtype,
                )
                pixel_values_flat = torch.cat([pixel_values_flat, padding], dim=0)
            elif num_real_tiles > compiled_batch:
                logger.warning(
                    f"Got {num_real_tiles} image tiles but vision encoder compiled for {compiled_batch}. Truncating."
                )
                pixel_values_flat = pixel_values_flat[:compiled_batch]
                num_real_tiles = compiled_batch

            image_features = self._traced_vision_encoder(pixel_values_flat)
            # Only use features from real tiles (discard padding outputs)
            image_features = image_features[:num_real_tiles]
            image_embeds, image_token_mask = self._build_image_injection_from_features(
                input_ids,
                image_features,
                image_embeds,
                image_token_mask,
            )

        return image_embeds, image_token_mask

    def _build_image_injection_from_features(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor,
        image_embeds: torch.Tensor,
        image_token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map flattened image features to ``image_token_id`` positions."""
        hidden_size = image_embeds.shape[-1]
        image_features_flat = image_features.reshape(-1, hidden_size)

        img_token_idx = 0
        for b in range(input_ids.shape[0]):
            positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=False).squeeze(-1)
            n = positions.shape[0]
            if n > 0:
                remaining = image_features_flat.shape[0] - img_token_idx
                if remaining <= 0:
                    logger.warning(
                        "No more image features available to inject; leaving remaining image tokens as text."
                    )
                    break
                take = min(n, remaining)
                selected_positions = positions[:take]
                image_embeds[b, selected_positions] = image_features_flat[img_token_idx : img_token_idx + take].to(
                    image_embeds.dtype
                )
                image_token_mask[b, selected_positions] = True
                if take < n:
                    logger.warning(
                        f"Image token count ({n}) exceeds available image features ({remaining}) for batch index {b}."
                    )
                img_token_idx += take
        return image_embeds, image_token_mask

    # ------------------------------------------------------------------
    # State dict helpers
    # ------------------------------------------------------------------

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: PretrainedConfig, neuron_config: NxDVLMNeuronConfig
    ) -> dict:
        # Remove vision model and connector weights (they live in a separate compiled artifact)
        keys_to_remove = [
            k for k in state_dict if k.startswith("model.vision_model.") or k.startswith("model.connector.")
        ]
        for k in keys_to_remove:
            del state_dict[k]

        text_config = getattr(config, "text_config", config)
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, text_config)

        # Add rank tensors required by NeuronAttentionBase
        tp_degree = neuron_config.tp_degree
        for i in range(text_config.num_hidden_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    # ------------------------------------------------------------------
    # Neuron config factory
    # ------------------------------------------------------------------

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        dtype: torch.dtype,
        prefill_chunk_size: int = 0,
        max_num_images: int = None,
    ) -> NxDVLMNeuronConfig:
        continuous_batching = (batch_size > 1) if batch_size else False
        config = AutoConfig.from_pretrained(checkpoint_id)
        vision_config = getattr(config, "vision_config", None)
        image_size = getattr(vision_config, "image_size", 512) if vision_config else 512
        patch_size = getattr(vision_config, "patch_size", 16) if vision_config else 16
        scale_factor = getattr(config, "scale_factor", 4)
        image_seq_len = (image_size // patch_size) ** 2 // (scale_factor**2)

        if max_num_images is None:
            # Auto-compute to support image splitting (tiling).
            # Idefics3 processors split images into tiles of ``image_size`` and add a global view.
            # With longest_edge=2048 (common default), the max grid is (2048/image_size)^2 + 1.
            max_tiles_per_dim = max(1, 2048 // image_size) if image_size > 0 else 4
            max_num_images = max_tiles_per_dim**2 + 1
            logger.info(f"Auto-computed max_num_images={max_num_images} for image_size={image_size}")
        return NxDVLMNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=dtype,
            target=instance_type,
            on_device_sampling=True,
            fused_qkv=True,
            continuous_batching=continuous_batching,
            prefill_chunk_size=prefill_chunk_size,
            max_num_images=max_num_images,
            image_size=image_size,
            image_seq_len=image_seq_len,
        )

    # ------------------------------------------------------------------
    # Export / load / save
    # ------------------------------------------------------------------

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig | None",
        neuron_config: NxDVLMNeuronConfig,
        token=None,
        revision=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        trust_remote_code=False,
        load_weights=False,
        **kwargs,
    ):
        from ....cache.entries.single_model import SingleModelCacheEntry
        from ....cache.hub_cache import hub_neuronx_cache
        from ....utils.instance import align_compilation_target

        if len(kwargs) > 0:
            logger.warning("Ignoring unsupported kwargs: %s", list(kwargs.keys()))

        compilation_target = align_compilation_target(neuron_config.target, override=False)
        if compilation_target != neuron_config.target:
            raise ValueError(
                f"Compilation target mismatch: neuron_config says {neuron_config.target!r} but "
                f"NEURON_PLATFORM_TARGET_OVERRIDE is {compilation_target!r}."
            )

        if config is None:
            # Keep the full VLM config — do NOT call get_text_config()
            config = AutoConfig.from_pretrained(
                model_id,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                trust_remote_code=trust_remote_code,
            )

        text_config = getattr(config, "text_config", config)
        config.torch_dtype = neuron_config.torch_dtype
        text_config.torch_dtype = neuron_config.torch_dtype
        if hasattr(text_config, "head_dim") and text_config.head_dim is None:
            text_config.head_dim = text_config.hidden_size // text_config.num_attention_heads

        # Compile the vision encoder first to avoid XLA/process-state conflicts
        # during subsequent decoder graph compilation in the same export flow.
        traced_vision_encoder = cls._compile_vision_encoder(model_id, config, neuron_config, token=token)

        # Compile the decoder graph bundle
        graph_builders = cls.create_graph_builders(config=config, neuron_config=neuron_config)
        cache_entry = (
            None
            if os.path.exists(model_id)
            else SingleModelCacheEntry(model_id, task=cls.task, config=config, neuron_config=neuron_config)
        )
        with hub_neuronx_cache(entry=cache_entry):
            traced_text_model = NxDPreTrainedModel.compile(
                neuron_config=neuron_config,
                graph_builders=graph_builders,
                compiler_args=cls.get_compiler_args(neuron_config),
            )

        model = cls(
            config=config,
            neuron_config=neuron_config,
            traced_model=traced_text_model,
            graph_builders=graph_builders,
            traced_vision_encoder=traced_vision_encoder,
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

    @classmethod
    def _from_pretrained(
        cls,
        model_id,
        config: PretrainedConfig,
        revision=None,
        token=None,
        cache_dir=None,
        force_download=False,
        local_files_only=False,
        **kwargs,
    ):
        # Use NeuronConfig (base class) so it dispatches to NxDVLMNeuronConfig via the registry
        from ....configuration_utils import NeuronConfig
        from ....utils.instance import current_instance_type
        from ....utils.system import get_available_cores

        if len(kwargs) > 0:
            logger.warning("Ignoring unsupported kwargs: %s", list(kwargs.keys()))

        neuron_config = NeuronConfig.from_pretrained(model_id)
        if neuron_config.target != current_instance_type():
            raise ValueError(
                f"The model was compiled for {neuron_config.target!r} but the current instance "
                f"type is {current_instance_type()!r}."
            )
        if get_available_cores() < neuron_config.tp_degree:
            raise ValueError(
                f"The model requires at least {neuron_config.tp_degree} Neuron cores but only "
                f"{get_available_cores()} are available."
            )

        model_dir = model_id
        if not os.path.exists(model_id):
            from tempfile import TemporaryDirectory

            from huggingface_hub import snapshot_download

            with TemporaryDirectory() as tmpdir:
                snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_dir=tmpdir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    allow_patterns=[
                        cls.COMPILED_MODEL_FILE_NAME,
                        VISION_ENCODER_FILE_NAME,
                    ],
                )
                return cls._from_pretrained(
                    tmpdir,
                    config,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )

        traced_text_model = torch.jit.load(os.path.join(model_dir, cls.COMPILED_MODEL_FILE_NAME))
        graph_builders = cls.create_graph_builders(config=config, neuron_config=neuron_config)
        vision_path = os.path.join(model_dir, VISION_ENCODER_FILE_NAME)
        traced_vision_encoder = None
        if os.path.exists(vision_path):
            traced_vision_encoder = torch.jit.load(vision_path)
            # Restore weights that were embedded by initialize_with_saved_weights during compilation.
            traced_vision_encoder.nxd_model.initialize_with_saved_weights(torch.tensor(0))

        model = cls(
            config=config,
            neuron_config=neuron_config,
            traced_model=traced_text_model,
            graph_builders=graph_builders,
            traced_vision_encoder=traced_vision_encoder,
        )

        # Load text decoder weights
        model.load_weights(
            model_dir,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
        return model

    def save(self, dest_path, weight_path=None):
        super().save(dest_path, weight_path)
        dest_path = normalize_path(dest_path)
        if self._traced_vision_encoder is not None:
            torch.jit.save(self._traced_vision_encoder, os.path.join(dest_path, VISION_ENCODER_FILE_NAME))
