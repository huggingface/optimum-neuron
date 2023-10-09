# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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


def check_neuron_model(neuron_model, batch_size=None, sequence_length=None, num_cores=None, auto_cast_type=None):
    neuron_config = getattr(neuron_model.config, "neuron", None)
    assert neuron_config
    if batch_size:
        assert neuron_config["batch_size"] == batch_size
    if sequence_length:
        assert neuron_config["sequence_length"] == sequence_length
    if num_cores:
        assert neuron_config["num_cores"] == num_cores
    if auto_cast_type:
        assert neuron_config["auto_cast_type"] == auto_cast_type
