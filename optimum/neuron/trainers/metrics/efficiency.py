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


class EfficiencyPlugin(MetricPlugin):
    """Plugin for calculating training efficiency metrics using inter-plugin communication."""

    def __init__(self):
        super().__init__(
            name="training_efficiency",
            requires_accumulation=False,
            depends_on=["forward_pass", "backward_pass", "optimizer_step", "total_step"],
        )

    def is_enabled(self, args: 'NeuronTrainingArguments') -> bool:
        """Enable if efficiency metrics are requested."""
        return args.enable_efficiency_metrics

    def calculate_realtime(self, window_stats: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """
        Calculate training efficiency from component timing metrics.

        Training efficiency = (forward + backward + optimizer time) / total time
        This measures how much of the total training time is spent on useful computation
        vs overhead (data loading, synchronization, etc.).
        """
        # Get component times from other metrics using inter-plugin communication
        forward_time = collector.get_metric_average_time("forward_pass")
        backward_time = collector.get_metric_average_time("backward_pass")
        optimizer_time = collector.get_metric_average_time("optimizer_step")
        total_time = collector.get_metric_average_time("total_step")

        if total_time <= 0:
            return {}

        compute_time = forward_time + backward_time + optimizer_time
        efficiency_percentage = (compute_time / total_time) * 100

        # Break down into component percentages
        forward_percentage = (forward_time / total_time) * 100
        backward_percentage = (backward_time / total_time) * 100
        optimizer_percentage = (optimizer_time / total_time) * 100
        unaccounted_percentage = 100 - efficiency_percentage  # Communication + overhead

        return {
            "train/training_efficiency": round(efficiency_percentage, 2),
            "train/forward_time_percent": round(forward_percentage, 2),
            "train/backward_time_percent": round(backward_percentage, 2),
            "train/optimizer_time_percent": round(optimizer_percentage, 2),
            "train/unaccounted_time_percent": round(unaccounted_percentage, 2),
        }

    def calculate_summary(self, summary_data: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """Calculate summary training efficiency metrics from all collected data."""
        # Get summary data from all component metrics
        forward_data = collector.summary_metrics.get("forward_pass", {})
        backward_data = collector.summary_metrics.get("backward_pass", {})
        optimizer_data = collector.summary_metrics.get("optimizer_step", {})
        total_data = collector.summary_metrics.get("total_step", {})

        if not all(
            [
                forward_data.get("step_times", []),
                backward_data.get("step_times", []),
                optimizer_data.get("step_times", []),
                total_data.get("step_times", []),
            ]
        ):
            return {}

        efficiency_values = []

        forward_times = forward_data["step_times"]
        backward_times = backward_data["step_times"]
        optimizer_times = optimizer_data["step_times"]
        total_times = total_data["step_times"]

        min_steps = min(len(forward_times), len(backward_times), len(optimizer_times), len(total_times))

        forward_percentages = []
        backward_percentages = []
        optimizer_percentages = []
        unaccounted_percentages = []

        for i in range(min_steps):
            forward_time = forward_times[i]
            backward_time = backward_times[i]
            optimizer_time = optimizer_times[i]
            total_time = total_times[i]

            if total_time > 0:
                compute_time = forward_time + backward_time + optimizer_time
                efficiency = (compute_time / total_time) * 100
                efficiency_values.append(efficiency)

                # Component percentages
                forward_percentages.append((forward_time / total_time) * 100)
                backward_percentages.append((backward_time / total_time) * 100)
                optimizer_percentages.append((optimizer_time / total_time) * 100)
                unaccounted_percentages.append(100 - efficiency)

        if efficiency_values:
            return {
                "summary/training_efficiency_avg": round(sum(efficiency_values) / len(efficiency_values), 2),
                "summary/forward_time_percent_avg": round(sum(forward_percentages) / len(forward_percentages), 2),
                "summary/backward_time_percent_avg": round(
                    sum(backward_percentages) / len(backward_percentages), 2
                ),
                "summary/optimizer_time_percent_avg": round(
                    sum(optimizer_percentages) / len(optimizer_percentages), 2
                ),
                "summary/unaccounted_time_percent_avg": round(
                    sum(unaccounted_percentages) / len(unaccounted_percentages), 2
                ),
            }

        return {}