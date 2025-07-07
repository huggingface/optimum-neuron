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
"""Custom AcceleratedOptimizer for Neuron."""

from typing import Optional

import torch
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import DistributedType

from .utils.dataclasses import NeuronDistributedType


import accelerate
import torch_xla.core.xla_model as xm
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

    accelerate.optimizer.xm = xm


@requires_neuronx_distributed
def allreduce_sequence_parallel_gradients(optimizer):
    """
    All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.

    Modified from megatron-lm:
    https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
    """
    from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region

    grads = []
    for param_group in optimizer.__getstate__()["param_groups"]:
        for group, params in param_group.items():
            if group == "params":
                for p in params:
                    if isinstance(p, torch.Tensor) and p.grad is not None:
                        sequence_parallel_param = getattr(p, "sequence_parallel_enabled", False)
                        if sequence_parallel_param:
                            grads.append(p.grad.data)
    for grad in grads:
        # sum v.s. average: sum
        reduce_from_tensor_model_parallel_region(grad)


class NeuronAcceleratedOptimizer(AcceleratedOptimizer):
    def __init__(
        self,
        optimizer: "torch.optim.Optimizer",
        device_placement: bool = True,
        scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    ):
        super().__init__(optimizer, device_placement=device_placement, scaler=scaler)

        self.parameters = []
        self.parameter_ids = {}
        self.clip_grad_norm_to_perform = None
        self.grad_norm = None
        if self.accelerator_state.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
            self.parameter_ids = {id(p) for p in self.parameters}

    # TODO: might be needed to override this soon.
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    def prepare_clip_grad_norm(self, parameters, max_norm, norm_type=2):
        self.clip_grad_norm_to_perform = {"parameters": parameters, "max_norm": max_norm, "norm_type": norm_type}

    @requires_neuronx_distributed
    def step(self, closure=None):
        from neuronx_distributed import parallel_layers
        from neuronx_distributed.parallel_layers.grads import bucket_allreduce_gradients

        if self.gradient_state.sync_gradients:
            # For sequence-parallel, we have to explicitly all-reduce the layernorm gradients.
            if self.accelerator_state.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
                allreduce_sequence_parallel_gradients(self.optimizer)

            if isinstance(self.optimizer, ZeroRedundancyOptimizer):
                if self.clip_grad_norm_to_perform is not None:
                    # `ZeroRedundancyOptimizer` does not allow to pass a norm type, it could be done but postponing for
                    # now.
                    self.optimizer.grad_clipping = True
                    self.optimizer.max_norm = self.clip_grad_norm_to_perform["max_norm"]
                else:
                    self.optimizer.grad_clipping = False
                self.optimizer.step(closure=closure)
                # Resetting everything.
                self.optimizer.grad_clipping = False
                self.clip_grad_norm_to_perform = None
            elif (
                self.accelerator_state.distributed_type is DistributedType.XLA
                or self.accelerator_state.distributed_type is NeuronDistributedType.MODEL_PARALLELISM
            ):
                if parallel_layers.parallel_state.get_data_parallel_size() > 1:
                    bucket_allreduce_gradients(xm._fetch_gradients(self.optimizer))
                if self.clip_grad_norm_to_perform is not None:
                    parameters = self.clip_grad_norm_to_perform.pop("parameters", None)
                    if parameters is not None:
                        self.grad_norm = parallel_layers.clip_grad_norm(parameters, **self.clip_grad_norm_to_perform)
                    self.clip_grad_norm_to_perform = None
                self.optimizer.step(closure=closure)
            elif self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer, closure)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                # If we reduced the loss scale, it means the optimizer step was skipped because of gradient overflow.
                self._is_overflow = scale_after < scale_before
            else:
                self.optimizer.step(closure)

    def __getstate__(self):
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }
