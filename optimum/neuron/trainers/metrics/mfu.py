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


class MFUPlugin(MetricPlugin):
    """Calculates Model FLOPS Utilization - how efficiently we're using the hardware."""

    def __init__(self):
        super().__init__(name=MetricNames.MFU, requires_accumulation=False)

    def is_enabled(self, args: NeuronTrainingArguments) -> bool:
        return args.enable_mfu_metrics

    def calculate_realtime(self, window_stats: dict, collector: "TrainingMetricsCollector") -> dict[str, float]:
        """MFU = actual FLOPS / peak FLOPS as a percentage."""
        if (
            not window_stats
            or collector.model_params is None
            or window_stats.get("total_time", 0) <= 0
            or window_stats.get("total_tokens", 0) == 0
        ):
            return {}

        total_tokens = window_stats["total_tokens"]
        total_time = window_stats["total_time"]

        if collector.seq_length is None:
            raise ValueError("Sequence length must be set in the collector to calculate MFU.")

        N = collector.model_params
        L, H, Q, T = collector.num_layers, collector.num_heads, collector.head_dim, collector.seq_length
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_iter = flops_per_token * total_tokens
        actual_flops_per_sec = flops_per_iter / total_time
        peak_flops_per_sec = collector.peak_tflops_per_core * 1e12
        mfu_pct = (actual_flops_per_sec / peak_flops_per_sec) * 100

        return {"train/mfu": round(mfu_pct, 2)}

    def calculate_summary(self, summary_data: dict, collector: "TrainingMetricsCollector") -> dict[str, float]:
        """Average MFU over the entire training run."""
        step_times = summary_data.get("step_times", [])
        tokens_per_step = summary_data.get("tokens_per_step", [])

        if not step_times or collector.model_params is None:
            return {}

        N = collector.model_params
        L, H, Q, T = collector.num_layers, collector.num_heads, collector.head_dim, collector.seq_length
        flops_per_token = 6 * N + 12 * L * H * Q * T

        mfu_values = []
        for tokens, time in zip(tokens_per_step, step_times):
            if time > 0 and tokens > 0:
                flops_per_iter = flops_per_token * tokens
                actual_flops_per_sec = flops_per_iter / time
                peak_flops_per_sec = collector.peak_tflops_per_core * 1e12
                mfu_pct = (actual_flops_per_sec / peak_flops_per_sec) * 100
                mfu_values.append(mfu_pct)

        if mfu_values:
            return {"summary/mfu_avg": sum(mfu_values) / len(mfu_values)}

        return {}
