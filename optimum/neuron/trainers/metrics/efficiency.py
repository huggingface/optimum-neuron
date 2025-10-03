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


class EfficiencyPlugin(MetricPlugin):
    """Calculates how much time is spent on useful computation vs overhead."""

    def __init__(self):
        super().__init__(
            name=MetricNames.EFFICIENCY,
            requires_accumulation=False,
            depends_on=[
                MetricNames.FORWARD_PASS,
                MetricNames.BACKWARD_PASS,
                MetricNames.OPTIMIZER_STEP,
                MetricNames.TOTAL_STEP,
            ],
        )

    def is_enabled(self, args: NeuronTrainingArguments) -> bool:
        return args.enable_efficiency_metrics

    def calculate_realtime(self, window_stats: dict, collector: TrainingMetricsCollector) -> dict[str, float]:
        """Efficiency = compute time / total time. Shows how much time we spend doing useful work."""
        # Get timing data from other plugins
        forward_time = collector.get_metric_average_time(MetricNames.FORWARD_PASS)
        backward_time = collector.get_metric_average_time(MetricNames.BACKWARD_PASS)
        optimizer_time = collector.get_metric_average_time(MetricNames.OPTIMIZER_STEP)
        total_time = collector.get_metric_average_time(MetricNames.TOTAL_STEP)

        if total_time <= 0:
            return {}

        compute_time = forward_time + backward_time + optimizer_time
        efficiency_pct = (compute_time / total_time) * 100

        # Break down by component
        forward_pct = (forward_time / total_time) * 100
        backward_pct = (backward_time / total_time) * 100
        optimizer_pct = (optimizer_time / total_time) * 100
        overhead_pct = 100 - efficiency_pct  # Communication, data loading, etc.

        return {
            "train/efficiency": round(efficiency_pct, 2),
            "train/forward_time_percent": round(forward_pct, 2),
            "train/backward_time_percent": round(backward_pct, 2),
            "train/optimizer_time_percent": round(optimizer_pct, 2),
            "train/overhead_time_percent": round(overhead_pct, 2),
        }

    def calculate_summary(self, summary_data: dict, collector: TrainingMetricsCollector) -> dict[str, float]:
        """Calculate average efficiency over the entire training run."""
        # Get all the step times from component metrics
        forward_data = collector.summary_metrics.get(MetricNames.FORWARD_PASS, {})
        backward_data = collector.summary_metrics.get(MetricNames.BACKWARD_PASS, {})
        optimizer_data = collector.summary_metrics.get(MetricNames.OPTIMIZER_STEP, {})
        total_data = collector.summary_metrics.get(MetricNames.TOTAL_STEP, {})

        forward_times = forward_data.get("step_times", [])
        backward_times = backward_data.get("step_times", [])
        optimizer_times = optimizer_data.get("step_times", [])
        total_times = total_data.get("step_times", [])

        if not all([forward_times, backward_times, optimizer_times, total_times]):
            return {}

        min_steps = min(len(forward_times), len(backward_times), len(optimizer_times), len(total_times))

        efficiency_values = []
        forward_pcts = []
        backward_pcts = []
        optimizer_pcts = []
        overhead_pcts = []

        for i in range(min_steps):
            total_time = total_times[i]
            if total_time <= 0:
                continue

            forward_time = forward_times[i]
            backward_time = backward_times[i]
            optimizer_time = optimizer_times[i]

            compute_time = forward_time + backward_time + optimizer_time
            efficiency = (compute_time / total_time) * 100

            efficiency_values.append(efficiency)
            forward_pcts.append((forward_time / total_time) * 100)
            backward_pcts.append((backward_time / total_time) * 100)
            optimizer_pcts.append((optimizer_time / total_time) * 100)
            overhead_pcts.append(100 - efficiency)

        if not efficiency_values:
            return {}

        def avg(values):
            return round(sum(values) / len(values), 2)

        return {
            "summary/efficiency_avg": avg(efficiency_values),
            "summary/forward_time_percent_avg": avg(forward_pcts),
            "summary/backward_time_percent_avg": avg(backward_pcts),
            "summary/optimizer_time_percent_avg": avg(optimizer_pcts),
            "summary/overhead_time_percent_avg": avg(overhead_pcts),
        }
