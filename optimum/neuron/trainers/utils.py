# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import torch
import torch_xla.core.xla_model as xm


class XLAPrefetchIterator:
    def __init__(self, examples: list[dict[str, torch.Tensor]], prefetch_size: int = 1):
        self.examples = examples
        self.prefetch_size = prefetch_size
        self.current_index = 0
        self.buffer = []
        self._prefetch()

    def _prefetch(self):
        while len(self.buffer) < self.prefetch_size and self.current_index < len(self.examples):
            example = self.examples[self.current_index]
            example_on_xla = {k: v.to(xm.xla_device()) for k, v in example.items()}
            self.buffer.append(example_on_xla)
            self.current_index += 1

    def __iter__(self):
        return self

    def __next__(self):
        if not self.buffer:
            raise StopIteration
        xm.mark_step()
        next_example = self.buffer.pop(0)
        self._prefetch()
        return next_example
