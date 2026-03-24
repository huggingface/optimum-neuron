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
"""Runtime wrappers for VLM decoder graphs."""

import logging
from typing import cast

import torch
import torch.nn.functional as F

from .decoder_wrappers import CHUNKED_PREFILL_MODEL_TAG, NxDDecoderWrapperForCausalLM


class NxDDecoderWrapperForImageTextToText(NxDDecoderWrapperForCausalLM):
    """Context decoder wrapper that passes image injection tensors to VLM graphs."""

    def _forward(self, input_ids, position_ids, seq_ids, sampling_params, image_embeds=None, image_token_mask=None):
        return self.model(input_ids, position_ids, seq_ids, sampling_params, image_embeds, image_token_mask)

    def forward(self, input_ids, position_ids, seq_ids, sampling_params, image_embeds=None, image_token_mask=None):
        input_ids, position_ids, seq_ids = self.convert_int64_to_int32(input_ids, position_ids, seq_ids)
        compiled_batch_size = cast(int, self.neuron_config.batch_size)

        if self.tag == CHUNKED_PREFILL_MODEL_TAG:
            # For chunked prefill, repeat the last token so padded positions are
            # true no-op overwrites for KV scatter, mirroring base decoder logic.
            chunk_size = self.neuron_config.prefill_chunk_size
            pad_length = chunk_size - input_ids.shape[1]
            if pad_length > 0:
                last_input_id = input_ids[:, -1:]
                last_pos = position_ids[:, -1:]
                input_ids = torch.cat([input_ids, last_input_id.expand(-1, pad_length)], dim=-1)
                position_ids = torch.cat([position_ids, last_pos.expand(-1, pad_length)], dim=-1)
                if image_embeds is not None:
                    last_embed = image_embeds[:, -1:, :]
                    image_embeds = torch.cat([image_embeds, last_embed.expand(-1, pad_length, -1)], dim=1)
                if image_token_mask is not None:
                    last_mask = image_token_mask[:, -1:]
                    image_token_mask = torch.cat([image_token_mask, last_mask.expand(-1, pad_length)], dim=1)
        else:
            # Context-encoding path: pad to the compiled max context length.
            pad_length = self.neuron_config.max_context_length - input_ids.shape[1]
            pad_token_id = getattr(self.config, "pad_token_id", None) or 0
            input_ids = F.pad(input_ids, (0, pad_length), "constant", pad_token_id)
            position_ids = F.pad(position_ids, (0, pad_length), "constant", 1)
            if image_embeds is not None and pad_length > 0:
                pad = torch.zeros(
                    image_embeds.shape[0],
                    pad_length,
                    image_embeds.shape[-1],
                    dtype=image_embeds.dtype,
                    device=image_embeds.device,
                )
                image_embeds = torch.cat([image_embeds, pad], dim=1)
            if image_token_mask is not None and pad_length > 0:
                mask_pad = torch.zeros(
                    image_token_mask.shape[0],
                    pad_length,
                    dtype=image_token_mask.dtype,
                    device=image_token_mask.device,
                )
                image_token_mask = torch.cat([image_token_mask, mask_pad], dim=1)

        input_batch_size = seq_ids.shape[0]
        if input_batch_size > self.neuron_config.max_batch_size:
            raise ValueError(
                f"Input batch size {input_batch_size} exceeds the maximum batch size "
                f"{self.neuron_config.max_batch_size}."
            )

        if input_batch_size == compiled_batch_size:
            return self._forward(input_ids, position_ids, seq_ids, sampling_params, image_embeds, image_token_mask)

        cur_batch = 0
        outputs = []
        args = (input_ids, position_ids, seq_ids, sampling_params, image_embeds, image_token_mask)

        logging.debug(
            f"get input_batch_size as {input_batch_size} but compiled batch_size as {self.neuron_config.batch_size}"
        )

        while cur_batch < input_batch_size:
            if cur_batch + compiled_batch_size <= input_batch_size:
                # Process full compiled-size chunks directly.
                chunk = [arg[cur_batch : cur_batch + compiled_batch_size] if arg is not None else None for arg in args]
                chunk_outputs = self._forward(*chunk)
            else:
                # Final partial chunk: pad up to compiled batch size.
                chunk = [arg[cur_batch:input_batch_size] if arg is not None else None for arg in args]
                chunk_outputs = self._forward_with_pad_vlm(*chunk)

            outputs.append(chunk_outputs)
            cur_batch += compiled_batch_size

        return torch.cat(outputs, dim=0)

    def _forward_with_pad_vlm(self, input_ids, position_ids, seq_ids, sampling_params, image_embeds, image_token_mask):
        compiled_batch_size = cast(int, self.neuron_config.batch_size)
        max_batch_size = cast(int, self.neuron_config.max_batch_size)

        def pad_to_batch(tensor):
            if tensor is None or tensor.shape[0] == compiled_batch_size:
                return tensor
            repeat_dims = [1] * (tensor.dim() - 1)
            padded = tensor[0].unsqueeze(0).repeat(compiled_batch_size, *repeat_dims).to(tensor.dtype)
            padded[: tensor.shape[0]] = tensor
            return padded

        seq_ids_list = seq_ids.tolist()
        padded_seq_ids = torch.tensor(
            seq_ids_list + [x for x in range(max_batch_size) if x not in seq_ids_list],
            dtype=seq_ids.dtype,
        )
        outputs = self._forward(
            pad_to_batch(input_ids),
            pad_to_batch(position_ids),
            padded_seq_ids,
            pad_to_batch(sampling_params),
            pad_to_batch(image_embeds),
            pad_to_batch(image_token_mask),
        )
        return outputs[: seq_ids.shape[0]]


class NxDTokenGenerationWrapperForImageTextToText(NxDDecoderWrapperForCausalLM):
    """Token generation wrapper that forwards dummy image-injection tensors."""

    def __init__(self, config, neuron_config, model, tag):
        super().__init__(config=config, neuron_config=neuron_config, model=model, tag=tag)
        # Runtime caches to avoid reallocating the same dummy tensors on every decode step.
        self._dummy_image_embeds = None
        self._dummy_image_token_mask = None

    def _forward(self, input_ids, position_ids, seq_ids, sampling_params):
        text_config = getattr(self.config, "text_config", self.config)
        model_dtype = cast(torch.dtype, self.neuron_config.torch_dtype or torch.float32)
        hidden_size = cast(int, text_config.hidden_size)
        embeds_shape = (input_ids.shape[0], input_ids.shape[1], hidden_size)
        mask_shape = (input_ids.shape[0], input_ids.shape[1])

        dummy_image_embeds = getattr(self, "_dummy_image_embeds", None)
        if (
            dummy_image_embeds is None
            or dummy_image_embeds.shape != embeds_shape
            or dummy_image_embeds.dtype != model_dtype
            or dummy_image_embeds.device != input_ids.device
        ):
            # The mask is always false in token generation, so image_embeds values are ignored.
            dummy_image_embeds = torch.empty(embeds_shape, dtype=model_dtype, device=input_ids.device)
            self._dummy_image_embeds = dummy_image_embeds

        dummy_image_token_mask = getattr(self, "_dummy_image_token_mask", None)
        if (
            dummy_image_token_mask is None
            or dummy_image_token_mask.shape != mask_shape
            or dummy_image_token_mask.dtype != torch.bool
            or dummy_image_token_mask.device != input_ids.device
        ):
            dummy_image_token_mask = torch.zeros(mask_shape, dtype=torch.bool, device=input_ids.device)
            self._dummy_image_token_mask = dummy_image_token_mask

        return self.model(
            input_ids,
            position_ids,
            seq_ids,
            sampling_params,
            dummy_image_embeds,
            dummy_image_token_mask,
        )
