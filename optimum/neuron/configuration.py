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
"""Configuration classes for Neuron Runtime."""

from ..configuration_utils import BaseConfig


class NeuronRTConfig(BaseConfig):
    """
    NeuronRTConfig is the configuration class handling all the Neuron Runtime parameters related to the Neuron compiled model,
    data parallel, Neuron core allocation and bucketization parameters.

    Attributes:
        dynamic_batch_size_unit (`Optional[int]`, defaults to `None`):
            Batch size used by neuron compiler when dynamic batch size support is on. The batch size of inputs should be
            multiple of this value.
        bucket_sizes (`Optional[Dict[List[Int]]]`, defaults to `None`):
            Input shape thresholds used for creating bucketed models.

    """

    CONFIG_NAME = "neuron_config.json"
    FULL_CONFIGURATION_FILE = "neuron_config.json"

    def __init__(
        self,
        dynamic_batch_size_unit: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.dynamic_batch_size_unit = dynamic_batch_size_unit

    @staticmethod
    def dataclass_to_dict(config) -> dict:
        new_config = {}
        if config is None:
            return new_config
        if isinstance(config, dict):
            return config
        for k, v in asdict(config).items():
            if isinstance(v, Enum):
                v = v.name
            elif isinstance(v, list):
                v = [elem.name if isinstance(elem, Enum) else elem for elem in v]
            new_config[k] = v
        return new_config
