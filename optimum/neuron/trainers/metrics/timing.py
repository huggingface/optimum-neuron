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

from ..training_args import NeuronTrainingArguments
from .base import MetricPlugin
from .collector import TrainingMetricsCollector
from .constants import MetricNames


class ComponentTimingPlugin(MetricPlugin):
    """Tracks individual component times (forward, backward, optimizer, total)."""

    def __init__(self):
        super().__init__(
            name="component_timing",
            requires_accumulation=True,  # forward/backward need accumulation across gradient steps
        )

    def is_enabled(self, args: NeuronTrainingArguments) -> bool:
        return True  # Always needed for efficiency calculations

    def get_metric_names(self) -> list[str]:
        return [
            MetricNames.FORWARD_PASS,
            MetricNames.BACKWARD_PASS,
            MetricNames.OPTIMIZER_STEP,
            MetricNames.TOTAL_STEP,
        ]

    def calculate_realtime(self, window_stats: dict, collector: TrainingMetricsCollector) -> dict[str, float]:
        """This plugin just provides timing data to other plugins."""
        return {}

    def calculate_summary(self, summary_data: dict, collector: TrainingMetricsCollector) -> dict[str, float]:
        """This plugin just provides timing data to other plugins."""
        return {}
