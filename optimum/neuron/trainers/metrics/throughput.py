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


class ThroughputPlugin(MetricPlugin):
    """Plugin for calculating training throughput metrics."""

    def __init__(self):
        super().__init__(name="throughput", requires_accumulation=False)

    def is_enabled(self, args: 'NeuronTrainingArguments') -> bool:
        """Enable if throughput metrics are requested."""
        return args.enable_throughput_metrics

    def calculate_realtime(self, window_stats: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """Calculate real-time throughput metrics from moving window."""
        if not window_stats or window_stats.get("total_time", 0) <= 0:
            return {}

        total_tokens = window_stats["total_tokens"]
        total_time = window_stats["total_time"]

        metrics = {}

        if total_tokens > 0:
            local_tokens_per_sec = total_tokens / total_time
            global_tokens_per_sec = local_tokens_per_sec * collector.dp_size
            metrics["train/global_tokens_per_sec"] = global_tokens_per_sec

        metrics["train/local_step_time"] = window_stats["avg_time_per_step"]

        return metrics

    def calculate_summary(self, summary_data: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """Calculate summary throughput metrics from all collected data."""
        if not summary_data.get("step_times"):
            return {}

        summary = {}
        step_times = summary_data["step_times"]
        tokens_per_step = summary_data["tokens_per_step"]

        local_tokens_per_sec_values = [
            tokens / time if time > 0 else 0 for tokens, time in zip(tokens_per_step, step_times)
        ]

        global_tokens_per_sec_values = [rate * collector.dp_size for rate in local_tokens_per_sec_values]
        if global_tokens_per_sec_values:
            summary.update(
                {
                    "summary/global_tokens_per_sec_avg": sum(global_tokens_per_sec_values)
                    / len(global_tokens_per_sec_values),
                }
            )

        summary.update(
            {
                "summary/total_training_steps": len(step_times),
                "summary/global_tokens_processed": sum(tokens_per_step) * collector.dp_size,
            }
        )

        return summary