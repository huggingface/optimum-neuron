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


class MFUPlugin(MetricPlugin):
    """Plugin for calculating Model FLOPS Utilization (MFU) metrics."""

    def __init__(self):
        super().__init__(name="mfu", requires_accumulation=False)

    def is_enabled(self, args: 'NeuronTrainingArguments') -> bool:
        """Enable if MFU metrics are requested."""
        return args.enable_mfu_metrics

    def calculate_realtime(self, window_stats: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """Calculate real-time MFU metrics from moving window."""
        if (
            not window_stats
            or collector.model_params is None
            or window_stats.get("total_time", 0) <= 0
            or window_stats.get("total_tokens", 0) == 0
        ):
            return {}

        total_tokens = window_stats["total_tokens"]
        total_time = window_stats["total_time"]

        # Theoretical FLOPs calculation for transformers:
        # Forward pass: ~6 * params * tokens, Backward pass: ~2 * forward pass FLOPs
        # Total: ~18 * params * tokens
        theoretical_flops = 18 * collector.model_params * total_tokens
        actual_flops_per_sec = theoretical_flops / total_time
        peak_flops_per_sec = collector.peak_tflops_per_core * 1e12 * collector.total_neuron_cores
        mfu_percentage = (actual_flops_per_sec / peak_flops_per_sec) * 100

        return {
            "train/mfu": round(mfu_percentage, 2),
        }

    def calculate_summary(self, summary_data: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """Calculate summary MFU metrics from all collected data."""
        if not summary_data.get("step_times") or collector.model_params is None:
            return {}

        step_times = summary_data["step_times"]
        tokens_per_step = summary_data["tokens_per_step"]

        mfu_values = []
        for tokens, t in zip(tokens_per_step, step_times):
            if t > 0 and tokens > 0:
                theoretical_flops = 18 * collector.model_params * tokens
                actual_flops_per_sec = theoretical_flops / t
                peak_flops_per_sec = collector.peak_tflops_per_core * 1e12 * collector.total_neuron_cores
                mfu_percentage = (actual_flops_per_sec / peak_flops_per_sec) * 100
                mfu_values.append(mfu_percentage)

        if mfu_values:
            return {
                "summary/mfu_avg": sum(mfu_values) / len(mfu_values),
            }

        return {}