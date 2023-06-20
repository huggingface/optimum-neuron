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

from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import DistributedType

from ..utils import is_torch_xla_available
from .utils.dataclasses import NeuronDistributedType


if is_torch_xla_available():
    import accelerate
    import torch_xla.core.xla_model as xm

    accelerate.optimizer.xm = xm


class NeuronAcceleratedOptimizer(AcceleratedOptimizer):
    # TODO: might be needed to override this soon.
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            if self.accelerator_state.distributed_type is DistributedType.TPU:
                optimizer_args = {"closure": closure} if closure is not None else {}
                xm.optimizer_step(self.optimizer, optimizer_args=optimizer_args)
            elif self.accelerator_state.distributed_type is NeuronDistributedType.XLA_FSDP:
                self.optimizer.step(closure)
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
