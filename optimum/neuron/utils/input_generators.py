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
"""Dummy input generation classes."""

from typing import Optional

import torch

from optimum.utils import (
    DTYPE_MAPPER,
    DummyAudioInputGenerator,
    DummyInputGenerator,
    NormalizedTextConfig,
    NormalizedVisionConfig,
)


class DummyBeamValuesGenerator(DummyInputGenerator):
    """
    Generates dummy beam search inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "beam_idx",
        "beam_scores",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        num_beams: int = 1,
        **kwargs,
    ):
        self.task = task
        self.num_beams = num_beams

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "beam_idx":
            return torch.arange(0, self.num_beams, dtype=DTYPE_MAPPER.pt(int_dtype))
        elif input_name == "beam_scores":
            return torch.zeros((self.num_beams,), dtype=DTYPE_MAPPER.pt(float_dtype))


class DummyMaskedPosGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = ("masked_pos", "bool_masked_pos")

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedVisionConfig,
        batch_size: int,
        **kwargs,
    ):
        self.task = task
        self.image_size = getattr(normalized_config, "image_size", None)
        self.patch_size = getattr(normalized_config, "patch_size", None)
        self.batch_size = batch_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        num_patches = (self.image_size // self.patch_size) ** 2
        masked_pos = torch.randint(low=0, high=2, size=(self.batch_size, num_patches))
        if input_name == "masked_pos":
            return masked_pos
        elif input_name == "bool_masked_pos":
            return masked_pos.bool()


class DummyControNetInputGenerator(DummyInputGenerator):
    SUPPORTED_INPUT_NAMES = (
        # ControlNet inputs
        "timestep",
        "encoder_hidden_states",  # depending on the hidden_size of text encoder
        "controlnet_cond",
        "conditioning_scale",
        # ControlNet outputs -> UNet inputs
        "down_block_additional_residuals",
        "mid_block_additional_residual",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        batch_size: int,
        sequence_length: Optional[int] = None,
        num_channels: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        vae_scale_factor: Optional[int] = None,
        encoder_hidden_size: Optional[int] = None,
        **kwargs,
    ):
        self.task = task
        self.normalized_config = normalized_config
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.vae_scale_factor = vae_scale_factor
        self.text_encoder_hidden_size = encoder_hidden_size

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "timestep":
            shape = [self.batch_size]
            return self.random_int_tensor(shape, max_value=999, framework=framework, dtype=int_dtype)
        elif input_name == "encoder_hidden_states":
            shape = (self.batch_size, self.sequence_length, self.text_encoder_hidden_size)
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        elif input_name == "controlnet_cond":
            num_channels = getattr(
                self.normalized_config, "conditioning_channels", 3
            )  # num_channels = 3 since `do_convert_rgb=True`
            shape = (
                self.batch_size,
                num_channels,
                self.height * self.vae_scale_factor,
                self.width * self.vae_scale_factor,
            )
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)
        elif input_name == "conditioning_scale":
            return torch.tensor([1.0])
        elif input_name == "down_block_additional_residuals":
            sample_shape = (self.batch_size, self.normalized_config.block_out_channels[0], self.height, self.width)
            sample = self.random_float_tensor(sample_shape, framework=framework, dtype=float_dtype)
            down_block_res_samples = (sample,)
            num_past_cross_attn_blocks = 0
            height = self.height
            width = self.width
            for idx, down_block_type in enumerate(self.normalized_config.down_block_types):
                res_samples = ()
                shape = (self.batch_size, self.normalized_config.block_out_channels[idx], height, width)
                for _ in range(self.normalized_config.layers_per_block):
                    res_samples += (self.random_float_tensor(shape, framework=framework, dtype=float_dtype),)
                if idx != len(self.normalized_config.down_block_types) - 1:
                    # add output of downsampler
                    num_past_cross_attn_blocks += 1
                    height = height // 2
                    width = width // 2
                    shape = (self.batch_size, self.normalized_config.block_out_channels[idx], height, width)
                    res_samples += (self.random_float_tensor(shape, framework=framework, dtype=float_dtype),)
                down_block_res_samples += res_samples
            return down_block_res_samples
        elif input_name == "mid_block_additional_residual":
            num_cross_attn_blocks = self.normalized_config.down_block_types.count("CrossAttnDownBlock2D")
            out_channels = self.normalized_config.block_out_channels[-1]
            shape = (
                self.batch_size,
                out_channels,
                self.height // 2**num_cross_attn_blocks,
                self.width // 2**num_cross_attn_blocks,
            )
            return self.random_float_tensor(shape, framework=framework, dtype=float_dtype)


# copied from https://github.com/huggingface/optimum/blob/171020c775cec6ff77826c3f5f5e5c1498b23f81/optimum/exporters/onnx/model_configs.py#L1363C1-L1368C111
class ASTDummyAudioInputGenerator(DummyAudioInputGenerator):
    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        shape = [self.batch_size, self.normalized_config.max_length, self.normalized_config.num_mel_bins]
        if input_name == "input_values":
            return self.random_float_tensor(shape, min_value=-1, max_value=1, framework=framework, dtype=float_dtype)
        return super().generate(input_name, framework=framework, int_dtype=int_dtype, float_dtype=float_dtype)
