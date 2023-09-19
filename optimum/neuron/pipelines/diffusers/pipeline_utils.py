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

import torch

class DiffusionPipelineMixin:
    def check_num_images_per_prompt(self, prompt_batch_size: int, neuron_batch_size: int, num_images_per_prompt: int):
        if self.dynamic_batch_size:
            return prompt_batch_size, num_images_per_prompt
        if neuron_batch_size != prompt_batch_size * num_images_per_prompt:
            raise ValueError(
                f"Models in the pipeline were compiled with `batch_size` {neuron_batch_size} which does not equal the number of"
                f" prompt({prompt_batch_size}) multiplied by `num_images_per_prompt`({num_images_per_prompt}). You need to enable"
                " `dynamic_batch_size` or precisely configure `num_images_per_prompt` during the compilation."
            )
        else:
            return prompt_batch_size, num_images_per_prompt
    
    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.run_safety_checker
    def run_safety_checker(self, image, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt")
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        return image, has_nsfw_concept