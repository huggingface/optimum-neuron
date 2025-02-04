# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# ==============================================================================
import torch


def init_neuron():
    return torch.ops.neuron._init_neuron()


def to_nc(tensor, ordinal=0):
    return torch.ops.neuron._to_neuron(tensor.contiguous(), ordinal)


def cpu(tensor):
    return torch.ops.neuron._from_neuron(tensor)


def slice(tensor, dim, start, end, step):
    return torch.ops.neuron._slice_neuron(tensor, dim, start, end, step)


def load(model, nc_id, nc_count):
    model.set_neuron_devices(nc_id, nc_count)
    return torch.ops.neuron._load_neuron(model)


def execute(model, inputs):
    return torch.ops.neuron._execute_neuron(model, inputs)


def parallel_to_nc(tensors):
    return torch.ops.neuron._parallel_to_neuron(tensors)


def parallel_cpu(tensor):
    return torch.ops.neuron._parallel_from_neuron(tensor)


def parallel_write(tensor, tensors):
    return torch.ops.neuron._parallel_write_neuron(tensor, tensors)


def parallel_slice(tensor, dim, start, end, step):
    return torch.ops.neuron._parallel_slice_neuron(tensor, dim, start, end, step)


def parallel_run(parallel_model, parallel_inputs, parallel_outputs):
    return torch.ops.neuron._parallel_run_neuron(parallel_model, parallel_inputs, parallel_outputs)


def parallel_profile_start(model, ntff_prefix, ntff_count_limit):
    return torch.ops.neuron._parallel_profile_start_neuron(model, ntff_prefix, ntff_count_limit)


def parallel_profile_stop(ntff_files):
    return torch.ops.neuron._parallel_profile_stop_neuron(ntff_files)
