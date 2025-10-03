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

from collections import deque


class MovingAverageWindow:
    """
    A moving average window for tracking metrics over a sliding window.

    Maintains separate deques for tokens, samples, and timing information,
    allowing for stable moving average calculations.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.tokens_per_step = deque(maxlen=window_size)
        self.samples_per_step = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)

    def add_step(self, tokens: int, samples: int, step_time: float):
        self.tokens_per_step.append(tokens)
        self.samples_per_step.append(samples)
        self.step_times.append(step_time)

    def get_window_stats(self) -> dict[str, float]:
        if not self.step_times:
            return {}

        total_tokens = sum(self.tokens_per_step)
        total_samples = sum(self.samples_per_step)
        total_time = sum(self.step_times)
        window_steps = len(self.step_times)

        return {
            "total_tokens": total_tokens,
            "total_samples": total_samples,
            "total_time": total_time,
            "window_steps": window_steps,
            "avg_tokens_per_step": total_tokens / window_steps if window_steps > 0 else 0,
            "avg_samples_per_step": total_samples / window_steps if window_steps > 0 else 0,
            "avg_time_per_step": total_time / window_steps if window_steps > 0 else 0,
        }

    def clear(self):
        self.tokens_per_step.clear()
        self.samples_per_step.clear()
        self.step_times.clear()

    @property
    def is_full(self) -> bool:
        return len(self.step_times) == self.window_size

    @property
    def size(self) -> int:
        return len(self.step_times)
