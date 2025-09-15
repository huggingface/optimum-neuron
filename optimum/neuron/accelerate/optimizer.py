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

import accelerate
import torch
import torch_xla.core.xla_model as xm
from accelerate.optimizer import AcceleratedOptimizer
from neuronx_distributed.parallel_layers.grads import bucket_allreduce_gradients, clip_grad_norm
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer


accelerate.optimizer.xm = xm


def allreduce_sequence_parallel_gradients(optimizer):
    """
    All-reduce layernorm parameters across model parallel nodes when sequence parallelism is used.

    Modified from megatron-lm:
    https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/blob/3f91f09bb2ab32f9904b47f46f19d2fc3f518ed8/megatron/training.py#L425
    """
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
        scaler: torch.amp.GradScaler | None = None,
    ):
        if scaler is not None:
            raise ValueError("NeuronAcceleratedOptimizer does not support `scaler`.")

        super().__init__(optimizer, device_placement=device_placement, scaler=scaler)

        self.clip_grad_norm_to_perform = None
        self.permanent_clip_grad_norm = None
        self.permanent_parameters = []
        self.grad_norm = None

    # TODO: might be needed to override this soon.
    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    def prepare_clip_grad_norm(self, parameters, max_norm, norm_type=2):
        self.clip_grad_norm_to_perform = {"parameters": parameters, "max_norm": max_norm, "norm_type": norm_type}

    def set_permanent_clip_grad_norm(self, max_norm, norm_type=2):
        self.permanent_clip_grad_norm = {"max_norm": max_norm, "norm_type": norm_type}
        self.permanent_parameters = []
        for param_group in self.optimizer.__getstate__()["param_groups"]:
            for group, params in param_group.items():
                if group == "params":
                    for p in params:
                        if isinstance(p, torch.Tensor) and p.requires_grad:
                            self.permanent_parameters.append(p)

    def _fetch_gradients(self):
        gradients = []
        ep_gradients = []
        for param_group in self.optimizer.__getstate__()["param_groups"]:
            for group, params in param_group.items():
                if group == "params":
                    for p in params:
                        if isinstance(p, torch.Tensor):
                            if p.grad is not None:
                                if hasattr(p, "expert_model_parallel") and p.expert_model_parallel:
                                    ep_gradients.append(p.grad.data)
                                else:
                                    gradients.append(p.grad.data)
                            elif hasattr(p, "main_grad"):
                                if hasattr(p, "expert_model_parallel") and p.expert_model_parallel:
                                    ep_gradients.append(p.main_grad.data)
                                else:
                                    gradients.append(p.main_grad.data)

        return gradients, ep_gradients

    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            # For sequence-parallel, we have to explicitly all-reduce the layernorm gradients.
            if self.accelerator_state.trn_config.sequence_parallel_enabled:
                allreduce_sequence_parallel_gradients(self.optimizer)

            if isinstance(self.optimizer, ZeroRedundancyOptimizer):
                if self.clip_grad_norm_to_perform is not None:
                    # `ZeroRedundancyOptimizer` does not allow to pass a norm type, it could be done but postponing for
                    # now.
                    self.optimizer.grad_clipping = True
                    self.optimizer.max_norm = self.clip_grad_norm_to_perform["max_norm"]
                elif self.permanent_clip_grad_norm is not None:
                    self.optimizer.grad_clipping = True
                    self.optimizer.max_norm = self.permanent_clip_grad_norm["max_norm"]
                else:
                    self.optimizer.grad_clipping = False
                self.optimizer.step(closure=closure)
                self.grad_norm = self.optimizer._grad_norm
                # Resetting everything.
                self.optimizer.grad_clipping = False
                self.clip_grad_norm_to_perform = None

            else:
                non_ep_gradients, ep_gradients = self._fetch_gradients()
                bucket_allreduce_gradients(non_ep_gradients + ep_gradients)
                if len(ep_gradients) > 0:
                    bucket_allreduce_gradients(non_ep_gradients, reduce_over_ep_group=True)

                if self.clip_grad_norm_to_perform is not None:
                    parameters = self.clip_grad_norm_to_perform.pop("parameters", None)
                    kwargs = self.clip_grad_norm_to_perform
                elif self.permanent_clip_grad_norm is not None:
                    parameters = self.permanent_parameters
                    kwargs = self.permanent_clip_grad_norm
                else:
                    parameters = None
                    kwargs = {}

                if parameters is not None:
                    self.grad_norm = clip_grad_norm(parameters, **kwargs)
                    self.clip_grad_norm_to_perform = None

                self.optimizer.step(closure=closure)

    def __getstate__(self):
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }
