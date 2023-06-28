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
"""Custom AcceleratedScheduler for Neuron."""

from accelerate.scheduler import AcceleratedScheduler

from .state import NeuronAcceleratorState


class NeuronAcceleratedScheduler(AcceleratedScheduler):
    def step(self, *args, **kwargs):
        if not self.step_with_optimizer:
            # No link between scheduler and optimizer -> just step
            self.scheduler.step(*args, **kwargs)
            return

        # Otherwise, first make sure the optimizer was stepped.
        if not self.gradient_state.sync_gradients:
            if self.gradient_state.adjust_scheduler:
                self.scheduler._step_count += 1
            return

        for opt in self.optimizers:
            if opt.step_was_skipped:
                return
        if self.split_batches:
            # Split batches -> the training dataloader batch size is not changed so one step per training step
            self.scheduler.step(*args, **kwargs)
        else:
            # Otherwise the training dataloader batch size was multiplied by `num_processes`, so we need to do
            # num_processes steps per training step
            num_processes = NeuronAcceleratorState().num_processes
            for _ in range(num_processes):
                # Special case when using OneCycle and `drop_last` was not used
                if hasattr(self.scheduler, "total_steps"):
                    if self.scheduler._step_count <= self.scheduler.total_steps:
                        self.scheduler.step(*args, **kwargs)
                else:
                    self.scheduler.step(*args, **kwargs)
