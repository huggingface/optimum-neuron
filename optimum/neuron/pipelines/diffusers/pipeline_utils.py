#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging

import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)


logger = logging.getLogger(__name__)


class NeuronStableDiffusionXLPipelineMixin:
    # Adapted from https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py#L573
    def _get_add_time_ids_text_to_image(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Adapted from https://github.com/huggingface/diffusers/blob/v0.21.4/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py#L582
    def _get_add_time_ids_image_to_image(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.get("requires_aesthetics_score"):
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    def _get_add_time_ids(self, *args, **kwargs):
        if self.auto_model_class in [StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline]:
            return self._get_add_time_ids_text_to_image(*args, **kwargs)
        elif self.auto_model_class in [StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline]:
            return self._get_add_time_ids_image_to_image(*args, **kwargs)
        else:
            raise ValueError(
                f"The pipeline type {self.auto_model_class} is not yet supported by Optimum Neuron, please open an request on: https://github.com/huggingface/optimum-neuron/issues."
            )
