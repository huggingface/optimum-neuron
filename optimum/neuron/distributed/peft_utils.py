# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities to enable distributed training with PEFT features."""

from typing import Literal, Optional, Union

from ..utils.require_utils import requires_peft


@requires_peft
def peft_tuner_linear_to_parallel_linear(
    tuner_layer: "BaseTunerLayer",
    axis: Union[Literal["row"], Literal["column"]],
    input_is_parallel: bool = False,
    gather_output: bool = True,
    stride: int = 1,
    linear_layer_weight_info: Optional[WeightInformation] = None,
    linear_layer_bias_weight_info: Optional[WeightInformation] = None,
    embedding_weight_to_tie: Optional["torch.nn.Parameter"] = None,
    sequence_parallel_enabled: bool = False,
    skip_weight_load: bool = False,
    device: Optional["torch.device"] = None,
) -> "BaseTunerLayer":
    from peft.tuners.tuners_utils import BaseTunerLayer

    # This is necessary for the case that the tuner layer wraps another tuner layer.
    parent = tuner_layer
    base_layer = tuner_layer
    while hasattr(base_layer, "base_layer"):
        parent = base_layer
        base_layer = base_layer.base_layer

    parallel_base_layer = linear_to_parallel_linear(
        base_layer,
        axis,
        input_is_parallel=input_is_parallel,
        gather_output=gather_output,
        stride=stride,
        linear_layer_weight_info=linear_layer_weight_info,
        linear_layer_bias_weight_info=linear_layer_bias_weight_info,
        embedding_weight_to_tie=embedding_weight_to_tie,
        sequence_parallel_enabled=sequence_parallel_enabled,
        skip_weight_load=skip_weight_load,
        device=device,
    )

    if isinstance(base_layer, BaseTunerLayer):
        tuner_layer = parallel_base_layer
    else:
        parent.base_layer = parallel_base_layer

    return tuner_layer
