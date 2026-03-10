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
"""Graph builders for vision-language model decoder graphs."""

import torch

from ..generation.sampling import prepare_sampling_params
from .decoder_builders import NxDDecoderBuilderForCausalLM


class NxDVLMContextEncodingBuilder(NxDDecoderBuilderForCausalLM):
    """Context encoding builder for VLMs.

    Extends the standard builder to include image-aware tensors as additional
    example inputs. The decoder computes text embeddings on-device, then injects
    ``image_embeds`` at positions marked by ``image_token_mask``.
    """

    def input_generator(self):
        input_ids = torch.zeros((self.neuron_config.batch_size, self.active_tokens), dtype=torch.int32)
        position_ids = torch.arange(self.active_tokens, dtype=torch.int32).expand(
            self.neuron_config.batch_size, self.active_tokens
        )
        seq_ids = torch.arange(0, self.neuron_config.batch_size, dtype=torch.int32)
        sampling_params_len = prepare_sampling_params(1).shape[1]
        sampling_params = torch.zeros((self.neuron_config.batch_size, sampling_params_len), dtype=torch.float32)
        text_config = getattr(self.config, "text_config", self.config)
        image_embeds = torch.zeros(
            (self.neuron_config.batch_size, self.active_tokens, text_config.hidden_size),
            dtype=self.neuron_config.torch_dtype,
        )
        image_token_mask = torch.zeros((self.neuron_config.batch_size, self.active_tokens), dtype=torch.bool)
        return [(input_ids, position_ids, seq_ids, sampling_params, image_embeds, image_token_mask)]


class NxDVLMTokenGenerationBuilder(NxDDecoderBuilderForCausalLM):
    """Token generation builder for VLMs.

    Extends the standard builder to include dummy image injection tensors so the
    compiled multi-graph model has a uniform signature matching the
    context encoding graph compiled by ``NxDVLMContextEncodingBuilder``.
    The dummy tensors are ignored at runtime.
    """

    def input_generator(self):
        base_inputs = super().input_generator()[0]
        input_ids, position_ids, seq_ids, sampling_params = base_inputs
        text_config = getattr(self.config, "text_config", self.config)
        dummy_image_embeds = torch.zeros(
            (self.neuron_config.batch_size, self.active_tokens, text_config.hidden_size),
            dtype=self.neuron_config.torch_dtype,
        )
        dummy_image_token_mask = torch.zeros((self.neuron_config.batch_size, self.active_tokens), dtype=torch.bool)
        return [(input_ids, position_ids, seq_ids, sampling_params, dummy_image_embeds, dummy_image_token_mask)]


class NxDVLMChunkedPrefillBuilder(NxDVLMContextEncodingBuilder):
    """Chunked prefill builder for VLMs.

    Uses the same image-injection signature as context encoding but with ``active_tokens``
    equal to ``prefill_chunk_size`` so runtime can prefill prompts in chunks while
    injecting image-aware features.
    """
