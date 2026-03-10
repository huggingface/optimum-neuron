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

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, Idefics3ForConditionalGeneration, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.idefics3.modeling_idefics3 import Idefics3Connector

from ..backend.config import NxDVLMNeuronConfig
from ..backend.modules.decoder.vlm_decoder import NxDModelForImageTextToText
from ..llama.modeling_llama import NxDLlamaModel, convert_state_dict_to_fused_qkv


logger = logging.getLogger("Neuron")


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
    delegating to the standard Llama-style layer construction.  Overrides
    :meth:`forward` to accept and inject image embeddings during context
    encoding / chunked prefill.
    """

    def __init__(self, config: PretrainedConfig, neuron_config: NxDVLMNeuronConfig):
        text_config = getattr(config, "text_config", config)
        super().__init__(text_config, neuron_config)

    def forward(
        self,
        input_ids,
        position_ids,
        seq_ids,
        sampling_params,
        image_embeds=None,
        image_token_mask=None,
    ):
        """VLM-aware forward: injects image features into text embeddings at context encoding time."""
        hidden_states = self.compute_input_embeddings(input_ids)
        if (
            self._is_context_encoding(input_ids.shape[-1])
            and image_embeds is not None
            and image_token_mask is not None
        ):
            mask = image_token_mask.to(torch.bool).unsqueeze(-1)
            hidden_states = torch.where(mask, image_embeds.to(hidden_states.dtype), hidden_states)
        return self._forward_from_embeddings(hidden_states, position_ids, seq_ids, sampling_params)


class SmolVLMNxDModelForImageTextToText(NxDModelForImageTextToText):
    """NxD model for SmolVLM (Idefics3) vision-language inference.

    Manages named compiled traced bundles:

        * ``model_vision.pt`` — vision encoder bundle (SigLIP vision encoder +
            HF ``Idefics3Connector``).
        * ``model_text.pt`` — decoder graph bundle(s) for context
            encoding / token generation (and speculation when enabled).

    At context/chunked-prefill time the decoder computes text embeddings on
    Neuron, then injects vision features at ``image_token_id`` positions.
    Token generation uses the standard ``input_ids`` path.
    """

    _model_cls = NxDSmolVLMDecoderModel
    _vision_encoder_cls = NeuronIdefics3VisionEncoder
    _STATE_DICT_MODEL_PREFIX = "model.text_model."
    task = "image-text-to-text"

    @classmethod
    def _get_vision_encoder_state_dict(cls, full_sd: dict) -> dict:
        """Extract and remap Idefics3 vision encoder weights from the full HF state dict."""
        sd = {}
        for k, v in full_sd.items():
            if k.startswith("model.vision_model."):
                sd["vision_model." + k[len("model.vision_model.") :]] = v
            elif k.startswith("model.connector."):
                sd["connector." + k[len("model.connector.") :]] = v
        return sd

    # ------------------------------------------------------------------
    # State dict helpers
    # ------------------------------------------------------------------

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: PretrainedConfig, neuron_config: NxDVLMNeuronConfig
    ) -> dict:
        # Remove vision model and connector weights (they are loaded through the VLM vision bundle).
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
    ) -> NxDVLMNeuronConfig:
        continuous_batching = (batch_size > 1) if batch_size else False
        config = AutoConfig.from_pretrained(checkpoint_id, revision=checkpoint_revision)
        if not hasattr(config, "vision_config"):
            raise ValueError(f"{checkpoint_id} does not have a vision_config; is it a VLM checkpoint?")
        vision_config = config.vision_config
        image_size = vision_config.image_size
        patch_size = vision_config.patch_size
        scale_factor = config.scale_factor
        image_seq_len = (image_size // patch_size) ** 2 // (scale_factor**2)

        # Auto-compute to support image splitting (tiling).
        # Idefics3 processors split images into tiles of ``image_size`` and add a global view.
        # With longest_edge=2048 (common default), the max grid is (2048/image_size)^2 + 1.
        max_tiles_per_dim = 2048 // image_size
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
