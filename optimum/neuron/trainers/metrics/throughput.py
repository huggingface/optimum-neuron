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

from __future__ import annotations

from typing import TYPE_CHECKING

from ..training_args import NeuronTrainingArguments
from .base import MetricPlugin
from .constants import MetricNames


if TYPE_CHECKING:
    from .collector import TrainingMetricsCollector


class ThroughputPlugin(MetricPlugin):
    """Calculates how many tokens/samples we process per second."""

    def __init__(self):
        super().__init__(name=MetricNames.THROUGHPUT, requires_accumulation=False)

    def is_enabled(self, args: NeuronTrainingArguments) -> bool:
        return args.enable_throughput_metrics

    def calculate_realtime(self, window_stats: dict, collector: "TrainingMetricsCollector") -> dict[str, float]:
        """Tokens per second across all devices."""
        if not window_stats or window_stats.get("total_time", 0) <= 0:
            return {}

        total_tokens = window_stats["total_tokens"]
        total_time = window_stats["total_time"]

        metrics = {}

        if total_tokens > 0:
            local_tps = total_tokens / total_time
            global_tps = local_tps * collector.dp_size  # Scale by number of data parallel workers
            metrics["train/tokens_per_sec"] = global_tps

        metrics["train/step_time"] = window_stats["avg_time_per_step"]
        return metrics

    def calculate_summary(self, summary_data: dict, collector: "TrainingMetricsCollector") -> dict[str, float]:
        """Average throughput over the entire training run."""
        step_times = summary_data.get("step_times", [])
        tokens_per_step = summary_data.get("tokens_per_step", [])

        if not step_times:
            return {}

        # Calculate tokens/sec for each step
        local_tps_values = [tokens / time if time > 0 else 0 for tokens, time in zip(tokens_per_step, step_times)]
        global_tps_values = [rate * collector.dp_size for rate in local_tps_values]

        summary = {}
        if global_tps_values:
            summary["summary/tokens_per_sec_avg"] = sum(global_tps_values) / len(global_tps_values)

        summary.update(
            {
                "summary/total_steps": len(step_times),
                "summary/total_tokens_processed": sum(tokens_per_step) * collector.dp_size,
            }
        )

        return summary
