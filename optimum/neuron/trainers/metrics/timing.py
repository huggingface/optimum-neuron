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

from typing import TYPE_CHECKING

from .base import MetricPlugin

if TYPE_CHECKING:
    from .collector import TrainingMetricsCollector
    from ..training_args import NeuronTrainingArguments


class ComponentTimingPlugin(MetricPlugin):
    """Plugin for handling individual timing component metrics (forward, backward, optimizer, total)."""

    def __init__(self):
        super().__init__(
            name="component_timing",
            requires_accumulation=True,  # forward_pass and backward_pass need accumulation
        )

    def is_enabled(self, args: 'NeuronTrainingArguments') -> bool:
        """Always enabled - needed for efficiency calculations."""
        return True

    def get_metric_names(self) -> list[str]:
        """Return all metric names this plugin handles."""
        return ["forward_pass", "backward_pass", "optimizer_step", "total_step"]

    def calculate_realtime(self, window_stats: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """
        Calculate real-time timing metrics.

        Note: This plugin doesn't produce its own train/ metrics,
        it just provides timing data for other plugins (like EfficiencyPlugin).
        """
        return {}

    def calculate_summary(self, summary_data: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """
        Calculate summary timing metrics.

        Note: This plugin doesn't produce its own summary/ metrics,
        it just provides timing data for other plugins (like EfficiencyPlugin).
        """
        return {}