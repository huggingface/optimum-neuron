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

from typing import TYPE_CHECKING, Optional

from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import DistributedType

from ..utils import is_neuronx_distributed_available, is_torch_xla_available
from .utils.dataclasses import NeuronDistributedType


if TYPE_CHECKING:
    import torch

if is_torch_xla_available():
    import accelerate
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer

    accelerate.optimizer.xm = xm

if is_neuronx_distributed_available():
    from neuronx_distributed import parallel_layers


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
        if self.accelerator_state.distributed_type is NeuronDistributedType.TENSOR_PARALLELISM:
            self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
            self.parameter_ids = {id(p) for p in self.parameters}

    # TODO: might be needed to override this soon.
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    def prepare_clip_grad_norm(self, parameters, max_norm, norm_type=2):
        parameter_ids = {id(p) for p in parameters}
        if parameter_ids == self.parameter_ids:
            self.clip_grad_norm_to_perform = {"max_norm": max_norm, "norm_type": norm_type}

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            if isinstance(self.optimizer, ZeroRedundancyOptimizer):
                if self.clip_grad_norm_to_perform is not None:
                    # `ZeroRedundancyOptimizer` does not allow to pass a norm type, it could be done but postponing for
                    # now.
                    self.optimizer.grad_clipping = True
                    self.optimizer.max_norm = self.clip_grad_norm_to_perform["max_norm"]
                else:
                    self.optimizer.grad_clipping = False
                optimizer_args = {"closure": closure} if closure is not None else {}
                self.optimizer.step(closure)
            elif self.accelerator_state.distributed_type is DistributedType.TPU:
                optimizer_args = {"closure": closure} if closure is not None else {}
                xm.optimizer_step(self.optimizer, optimizer_args=optimizer_args)
            elif self.accelerator_state.distributed_type is NeuronDistributedType.XLA_FSDP:
                self.optimizer.step(closure)
            elif self.accelerator_state.distributed_type is NeuronDistributedType.TENSOR_PARALLELISM:
                xm.reduce_gradients(
                    self.optimizer, groups=parallel_layers.parallel_state.get_data_parallel_group(as_list=True)
                )
                if self.clip_grad_norm_to_perform is not None:
                    parallel_layers.clip_grad_norm(self.parameters, **self.clip_grad_norm_to_perform)
                self.optimizer.step()
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
