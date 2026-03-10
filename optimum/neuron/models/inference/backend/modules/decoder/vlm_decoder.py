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
import os
from functools import partial
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, PretrainedConfig

from ......cache.entries.single_model import SingleModelCacheEntry
from ......cache.hub_cache import hub_neuronx_cache
from ......configuration_utils import NeuronConfig
from ......utils.instance import align_compilation_target, current_instance_type
from ......utils.system import get_available_cores
from ...config import NEURON_CONFIG_FILE, NxDVLMNeuronConfig
from ...pretrained_model import (
    NxDPreTrainedModel,
    NxDTracedModel,
    get_compiled_model_allow_patterns,
    list_compiled_model_paths,
)
from ..checkpoint import load_state_dict
from ..generation.sampling import validate_sampling_params
from .modeling_decoder import NxDModelForCausalLM
from .vlm_builders import (
    NxDVisionEncoderBuilder,
    NxDVLMChunkedPrefillBuilder,
    NxDVLMContextEncodingBuilder,
    NxDVLMTokenGenerationBuilder,
)
from .vlm_wrappers import NxDVLMContextDecoderWrapper, NxDVLMTokenGenerationWrapper


logger = logging.getLogger("Neuron")


class NxDVLMModelForCausalLM(NxDModelForCausalLM):
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
    * ``_export`` — full export flow (vision then decoder compilation).
    * ``convert_hf_to_neuron_state_dict`` — model-specific weight remapping.
    * ``_get_neuron_config`` — model-specific :class:`NxDVLMNeuronConfig` factory.
    """

    _vision_encoder_cls = None
    _context_wrapper_cls = NxDVLMContextDecoderWrapper
    _chunked_prefill_wrapper_cls = NxDVLMContextDecoderWrapper
    _token_generation_wrapper_cls = NxDVLMTokenGenerationWrapper

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
        traced_models: list[NxDTracedModel],
    ):
        if len(traced_models) < 2:
            raise ValueError("VLM models require at least two traced bundles: vision and text")
        super().__init__(config, neuron_config, traced_models)
        self.image_token_id = getattr(config, "image_token_id", None)

    def _get_vision_traced_model(self) -> torch.jit.ScriptModule:
        return self._traced_models[0].traced_model

    def _get_text_traced_model(self) -> torch.jit.ScriptModule:
        return self._traced_models[1].traced_model

    def _checkpoint_loader_for_bundle(self, model_index: int, checkpoint_path: str):
        if model_index == 0:
            return partial(self._vision_checkpoint_loader_fn, checkpoint_path, self.config, self.neuron_config)
        return partial(self.checkpoint_loader_fn, checkpoint_path, self.config, self.neuron_config)

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

        if pixel_values is not None and self.image_token_id is not None:
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

            image_features = self._get_vision_traced_model()(pixel_values_flat)
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
    # Save / load
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
        vision_graph_builders = cls.create_vision_graph_builders(config=config, neuron_config=neuron_config)
        logger.info("Compiling vision encoder with deferred weights …")
        traced_vision_encoder = NxDPreTrainedModel.compile(
            neuron_config=neuron_config,
            graph_builders=vision_graph_builders,
            compiler_args=cls.get_compiler_args(neuron_config),
        )

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
            traced_models=[
                NxDTracedModel(
                    traced_model=traced_vision_encoder,
                    graph_builders=vision_graph_builders,
                ),
                NxDTracedModel(traced_model=traced_text_model, graph_builders=graph_builders),
            ],
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
                        *get_compiled_model_allow_patterns(),
                        NEURON_CONFIG_FILE,
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

        compiled_paths = list_compiled_model_paths(model_dir)
        if len(compiled_paths) < 2:
            raise FileNotFoundError(
                f"Expected at least two compiled VLM artifacts (vision + text) under {model_dir}, found {len(compiled_paths)}"
            )
        traced_models_loaded = [torch.jit.load(path) for path in compiled_paths]
        vision_graph_builders = cls.create_vision_graph_builders(config=config, neuron_config=neuron_config)
        graph_builders = cls.create_graph_builders(config=config, neuron_config=neuron_config)

        model = cls(
            config=config,
            neuron_config=neuron_config,
            traced_models=[
                NxDTracedModel(traced_model=traced_models_loaded[0], graph_builders=vision_graph_builders),
                *[
                    NxDTracedModel(traced_model=traced_text_model, graph_builders=graph_builders)
                    for traced_text_model in traced_models_loaded[1:]
                ],
            ],
        )

        # Load all bundle weights (vision + text)
        model.load_weights(
            model_dir,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
        )
        return model
