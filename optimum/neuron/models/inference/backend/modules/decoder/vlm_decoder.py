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
"""Base class for vision-language model (VLM) decoder graph inference on Neuron (NxD)."""

import logging

import torch
from transformers import PretrainedConfig

from ...config import NxDVLMNeuronConfig
from ..checkpoint import load_state_dict
from ..generation.sampling import validate_sampling_params
from .modeling_decoder import NxDModelForCausalLM
from .vlm_builders import (
    NxDChunkedPrefillBuilderForImageTextToText,
    NxDDecoderBuilderForImageTextToText,
    NxDTokenGenerationBuilderForImageTextToText,
    NxDVisionEncoderBuilder,
)
from .vlm_wrappers import NxDDecoderWrapperForImageTextToText, NxDTokenGenerationWrapperForImageTextToText


logger = logging.getLogger("Neuron")


class NxDModelForImageTextToText(NxDModelForCausalLM):
    """Base class for NxD vision-language model (VLM) inference.

    Extends :class:`NxDModelForCausalLM` with:

        * Indexed compiled artifacts where bundle 0 is the vision encoder and
            bundle 1+ are decoder graphs.
    * VLM-aware graph wrappers for context encoding, chunked prefill, and token
      generation (the compiled decoder graph always receives ``image_embeds`` and
      ``image_token_mask`` arguments so that context and token-generation graphs
      share a uniform signature).
    * ``pixel_values`` capture in :meth:`generate` and image-token injection
      during context encoding / chunked prefill.

    Subclasses must set:
    * ``_model_cls`` — the inner decoder ``nn.Module`` (e.g. a Llama-style model).
    * ``_vision_encoder_cls`` — the vision encoder ``nn.Module`` to compile.

    Subclasses must implement:
    * ``_get_vision_encoder_state_dict`` — remap raw HF weights to the vision
      encoder's key namespace.
    * ``convert_hf_to_neuron_state_dict`` — model-specific weight remapping.
    * ``_get_neuron_config`` — model-specific :class:`NxDVLMNeuronConfig` factory.
    """

    _vision_encoder_cls = None
    _text_bundle_key = "text"
    _context_wrapper_cls = NxDDecoderWrapperForImageTextToText
    _chunked_prefill_wrapper_cls = NxDDecoderWrapperForImageTextToText
    _token_generation_wrapper_cls = NxDTokenGenerationWrapperForImageTextToText

    @classmethod
    def _get_vision_encoder_state_dict(cls, full_sd: dict) -> dict:
        """Remap raw HF state-dict keys to this vision encoder's key namespace.

        Subclasses must override this to extract and rename the vision-encoder
        weights from the full model state dict loaded from disk.

        Args:
            full_sd: Full raw state dict loaded from the HF checkpoint on disk.

        Returns:
            State dict keyed for :attr:`_vision_encoder_cls`.
        """
        raise NotImplementedError(f"{cls.__name__} must implement _get_vision_encoder_state_dict")

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDVLMNeuronConfig,
        traced_models: dict[str, torch.jit.ScriptModule],
        graph_builders: dict[str, dict],
    ):
        if "vision" not in traced_models or "text" not in traced_models:
            raise ValueError("VLM models require 'vision' and 'text' bundles in traced_models")
        super().__init__(config, neuron_config, traced_models, graph_builders)
        self.image_token_id = getattr(config, "image_token_id", None)
        self._reset_prefill_state()

    def _reset_prefill_state(self):
        """Reset pre-computed image injection state to defaults."""
        self._current_pixel_values = None
        self._full_image_embeds = None
        self._full_image_token_mask = None
        self._prefill_chunk_offset = 0

    def _get_vision_traced_model(self) -> torch.jit.ScriptModule:
        return self._traced_models["vision"]

    def _get_text_traced_model(self) -> torch.jit.ScriptModule:
        return self._traced_models["text"]

    def get_checkpoint_loader_fn(self, bundle_name: str):
        if bundle_name == "vision":
            return self._vision_checkpoint_loader_fn
        return self.checkpoint_loader_fn

    @classmethod
    def create_vision_graph_builders(cls, config, neuron_config):
        vision_encoder_cls = cls._vision_encoder_cls
        if vision_encoder_cls is None:
            raise ValueError(f"{cls.__name__} must set _vision_encoder_cls before creating vision graph builders")
        return {
            "vision_encoder": NxDVisionEncoderBuilder(
                config=config,
                neuron_config=neuron_config,
                vision_encoder_cls=vision_encoder_cls,
                priority_model_idx=0,
            )
        }

    def _vision_checkpoint_loader_fn(self, checkpoint_path, config, neuron_config):
        full_sd = load_state_dict(checkpoint_path)
        model_sd = self._get_vision_encoder_state_dict(full_sd)
        if neuron_config.torch_dtype != torch.float32:
            for name, param in model_sd.items():
                if torch.is_floating_point(param) and param.dtype is not neuron_config.torch_dtype:
                    logger.debug(f"Converting {name} to {neuron_config.torch_dtype}")
                    model_sd[name] = param.to(neuron_config.torch_dtype)
        return model_sd

    # ------------------------------------------------------------------
    # Graph builder / compilation
    # ------------------------------------------------------------------

    @classmethod
    def create_graph_builders(cls, config, neuron_config):
        vision_builders = cls.create_vision_graph_builders(config=config, neuron_config=neuron_config)

        tkg_neuron_config = NxDModelForCausalLM._create_token_generation_config(neuron_config)
        text_builders = {}
        if neuron_config.prefill_chunk_size > 0:
            chunk_neuron_config = NxDModelForCausalLM._create_chunked_prefill_config(neuron_config)
            text_builders["chunked_prefill"] = NxDChunkedPrefillBuilderForImageTextToText(
                config=config,
                neuron_config=chunk_neuron_config,
                max_tokens=chunk_neuron_config.sequence_length,
                active_tokens=chunk_neuron_config.prefill_chunk_size,
                model_cls=cls._model_cls,
            )
        else:
            ctx_neuron_config = NxDModelForCausalLM._create_context_encoding_config(neuron_config)
            text_builders["context_encoding"] = NxDDecoderBuilderForImageTextToText(
                config=config,
                neuron_config=ctx_neuron_config,
                max_tokens=ctx_neuron_config.max_context_length,
                active_tokens=ctx_neuron_config.max_context_length,
                model_cls=cls._model_cls,
            )

        text_builders["token_generation"] = NxDTokenGenerationBuilderForImageTextToText(
            config=config,
            neuron_config=tkg_neuron_config,
            max_tokens=tkg_neuron_config.sequence_length,
            active_tokens=1,
            model_cls=cls._model_cls,
            priority_model_idx=0,
        )
        return {"vision": vision_builders, "text": text_builders}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def prepare_vlm_prefill(self, input_ids: torch.Tensor, pixel_values: torch.Tensor | None):
        """Pre-compute image injection tensors for chunked prefill.

        Could be called externally (e.g.: by the vLLM model wrapper) before processing chunks,
        and internally by :meth:`generate` before delegating to the base
        sampling loop.
        """
        image_features = self._encode_images(pixel_values)
        image_embeds, image_token_mask = self._prepare_image_injection_tensors(input_ids, image_features)
        self._full_image_embeds = image_embeds
        self._full_image_token_mask = image_token_mask
        self._prefill_chunk_offset = 0

    def generate(self, input_ids, attention_mask=None, pixel_values=None, generation_config=None, **kwargs):
        """Override generate to capture pixel_values for the context encoding forward pass.

        The base ``_sample()`` method does not pass ``pixel_values`` through to ``forward()``.
        We pre-compute full-sequence image embeddings here so that ``forward()``
        and ``prefill_chunk()`` can simply slice them per chunk without tracking
        a running feature offset.
        """
        self._current_pixel_values = pixel_values
        self.prepare_vlm_prefill(input_ids, pixel_values)
        try:
            return super().generate(
                input_ids, attention_mask=attention_mask, generation_config=generation_config, **kwargs
            )
        finally:
            self._reset_prefill_state()

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        seq_ids: torch.LongTensor = None,
        sampling_params: torch.FloatTensor = None,
        pixel_values: torch.FloatTensor = None,
    ):
        # Retrieve pixel_values stored by generate() if not passed directly
        if pixel_values is None:
            pixel_values = getattr(self, "_current_pixel_values", None)

        if self.neuron_config.on_device_sampling:
            validate_sampling_params(sampling_params, self.neuron_config.max_topk)

        is_context_encoding = input_ids.shape[-1] > 1 and not position_ids.min().item()
        is_speculation = input_ids.shape[-1] == self.neuron_config.speculation_length

        if is_context_encoding:
            # Reuse tensors pre-computed by generate() when available;
            # otherwise encode and prepare now (direct forward() call).
            if self._full_image_embeds is not None:
                image_embeds = self._full_image_embeds
                image_token_mask = self._full_image_token_mask
            else:
                image_features = self._encode_images(pixel_values)
                image_embeds, image_token_mask = self._prepare_image_injection_tensors(input_ids, image_features)
            if self.neuron_config.prefill_chunk_size > 0:
                chunk_size = self.neuron_config.prefill_chunk_size
                outputs = None
                for chunk_start in range(0, input_ids.shape[-1], chunk_size):
                    chunk_end = min(chunk_start + chunk_size, input_ids.shape[-1])
                    chunk_ids = input_ids[:, chunk_start:chunk_end]
                    chunk_pos = position_ids[:, chunk_start:chunk_end]
                    outputs = self.chunked_prefill_model(
                        chunk_ids,
                        chunk_pos,
                        seq_ids,
                        sampling_params,
                        image_embeds[:, chunk_start:chunk_end, :],
                        image_token_mask[:, chunk_start:chunk_end],
                    )
            else:
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
        """Process one prompt chunk while injecting image features on-device.

        Called once per chunk by ``generation_utils._sample()`` and by the vLLM
        runner.  Full-sequence image injection tensors are pre-computed by
        :meth:`generate` and stored on ``self``; this method simply slices them
        for the current chunk.
        """
        offset = self._prefill_chunk_offset
        chunk_len = input_ids.shape[-1]

        if self._full_image_embeds is not None:
            image_embeds = self._full_image_embeds[:, offset : offset + chunk_len, :]
            image_token_mask = self._full_image_token_mask[:, offset : offset + chunk_len]
        else:
            text_config = getattr(self.config, "text_config", self.config)
            image_embeds = torch.zeros(
                (input_ids.shape[0], chunk_len, text_config.hidden_size),
                dtype=self.neuron_config.torch_dtype,
                device=input_ids.device,
            )
            image_token_mask = torch.zeros(
                (input_ids.shape[0], chunk_len),
                dtype=torch.bool,
                device=input_ids.device,
            )
        self._prefill_chunk_offset = offset + chunk_len

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

    def _encode_images(self, pixel_values: torch.Tensor | None) -> torch.Tensor | None:
        """Run the vision encoder once and return image features.

        Args:
            pixel_values: Raw pixel values ``[B, N, C, H, W]`` or ``[B, C, H, W]``,
                or *None* when there are no images.

        Returns:
            Image features tensor ``[N_real_tiles, num_patches, H]`` with padding
            tiles discarded, or *None* if no images were provided.
        """
        if pixel_values is None or self.image_token_id is None:
            return None

        if pixel_values.dim() == 5:
            pixel_values_flat = pixel_values.reshape(-1, *pixel_values.shape[2:])
        else:
            pixel_values_flat = pixel_values
        pixel_values_flat = pixel_values_flat.to(self.neuron_config.torch_dtype)

        # Pad to the compiled vision encoder batch size (batch_size * max_num_images).
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

        image_features = self._get_vision_traced_model()(pixel_values_flat)
        # Only use features from real tiles (discard padding outputs)
        return image_features[:num_real_tiles]

    def _prepare_image_injection_tensors(
        self,
        input_ids: torch.Tensor,
        image_features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build image injection tensors from pre-computed image features.

        Args:
            input_ids: Token ids for the full prompt sequence.
            image_features: Pre-computed vision encoder output from
                :meth:`_encode_images`, or *None* when there are no images.

        Returns:
            A tuple ``(image_embeds, image_token_mask)`` for the full sequence.
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

        if image_features is not None:
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
