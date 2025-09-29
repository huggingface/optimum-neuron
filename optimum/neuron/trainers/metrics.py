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

import time
from typing import Any

import torch
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_size,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)

from ..utils.training_utils import get_model_param_count


class TrainingMetricsCollector:
    """
    Collects and calculates training performance metrics for Neuron distributed training.

    Provides both per-neuron-core metrics (for hardware utilization analysis) and
    general throughput metrics (for training performance comparison).
    """

    def __init__(self, model, training_args):
        self.model = model
        self.args = training_args

        # Hardware topology
        self.dp_size = get_data_parallel_size()
        self.tp_size = get_tensor_model_parallel_size()
        self.pp_size = get_pipeline_model_parallel_size()
        self.total_neuron_cores = self.dp_size * self.tp_size * self.pp_size

        # Model parameters (cached for MFU calculation)
        self.model_params = None
        if self.args.enable_mfu_metrics:
            self.model_params = get_model_param_count(model, trainable_only=False)

        # Hardware specs (TFLOPS per core for bf16)
        # Trainium v1: ~100 TFLOPS bf16 per core
        # Can be configured for different hardware generations
        self.peak_tflops_per_core = getattr(self.args, 'peak_tflops_per_core', 100.0)

        # Metrics collection state
        self.reset_window()

    def reset_window(self):
        """Reset the metrics collection window."""
        self.window_start_time = None
        self.window_start_step = None
        self.total_tokens_processed = 0
        self.total_samples_processed = 0
        self.step_count = 0

    def start_timing_window(self, step: int):
        """Start a new timing window for metrics collection."""
        self.window_start_time = time.time()
        self.window_start_step = step
        self.total_tokens_processed = 0
        self.total_samples_processed = 0
        self.step_count = 0

    def record_batch_metrics(self, inputs: dict[str, Any], step: int):
        """
        Record metrics for a single batch.

        Args:
            inputs: Batch inputs containing 'input_ids' for token counting
            step: Current training step
        """
        if not self.args.enable_throughput_metrics:
            return

        # Initialize window if needed
        if self.window_start_time is None:
            self.start_timing_window(step)

        # Count tokens and samples in this batch
        batch_tokens = 0
        batch_samples = 0

        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                batch_tokens = input_ids.numel()
                batch_samples = input_ids.size(0)

        # Accumulate across the window
        self.total_tokens_processed += batch_tokens
        self.total_samples_processed += batch_samples
        self.step_count += 1

    def calculate_throughput_metrics(self, elapsed_time: float) -> dict[str, float]:
        """Calculate throughput metrics for the current window."""
        if elapsed_time <= 0 or not self.args.enable_throughput_metrics:
            return {}

        metrics = {}

        # General throughput metrics (for training performance comparison)
        if self.total_tokens_processed > 0:
            metrics["tokens_per_sec"] = self.total_tokens_processed / elapsed_time
        if self.total_samples_processed > 0:
            metrics["samples_per_sec"] = self.total_samples_processed / elapsed_time

        # Per-neuron-core metrics (for hardware utilization analysis)
        if self.total_neuron_cores > 0:
            if self.total_tokens_processed > 0:
                metrics["tokens_per_sec_per_neuron_core"] = self.total_tokens_processed / (elapsed_time * self.total_neuron_cores)
            if self.total_samples_processed > 0:
                metrics["samples_per_sec_per_neuron_core"] = self.total_samples_processed / (elapsed_time * self.total_neuron_cores)

        return metrics

    def calculate_mfu_metrics(self, elapsed_time: float) -> dict[str, float]:
        """
        Calculate Model FLOPs Utilization (MFU) metrics.

        MFU measures what percentage of the peak hardware FLOPs are being utilized
        for useful computation (forward + backward pass).
        """
        if (not self.args.enable_mfu_metrics or
            elapsed_time <= 0 or
            self.model_params is None or
            self.total_tokens_processed == 0):
            return {}

        # Theoretical FLOPs calculation for transformers
        # Forward pass: ~6 * params * tokens (rough approximation)
        # Backward pass: ~2 * forward pass FLOPs
        # Total: ~18 * params * tokens
        theoretical_flops = 18 * self.model_params * self.total_tokens_processed

        # Actual FLOPs per second achieved
        actual_flops_per_sec = theoretical_flops / elapsed_time

        # Peak FLOPs available across all cores
        peak_flops_per_sec = self.peak_tflops_per_core * 1e12 * self.total_neuron_cores

        # MFU as percentage
        mfu_percentage = (actual_flops_per_sec / peak_flops_per_sec) * 100

        return {
            "model_flops_utilization": round(mfu_percentage, 2),
            "theoretical_flops_per_sec": actual_flops_per_sec,
            "peak_flops_per_sec": peak_flops_per_sec,
        }

    def calculate_efficiency_metrics(self, elapsed_time: float) -> dict[str, float]:
        """
        Calculate training efficiency metrics.

        Training efficiency measures how close we are to theoretical maximum throughput
        considering memory bandwidth, kernel efficiency, and communication overhead.
        """
        if not self.args.enable_efficiency_metrics or elapsed_time <= 0:
            return {}

        # For now, we can use a simplified efficiency metric
        # This could be expanded to include more sophisticated analysis
        metrics = {}

        # Simple efficiency based on tokens per second per core vs expected
        # This is a placeholder - in practice, you'd want to establish baselines
        # based on model size, sequence length, and hardware characteristics
        if "tokens_per_sec_per_neuron_core" in self.calculate_throughput_metrics(elapsed_time):
            tokens_per_core = self.total_tokens_processed / (elapsed_time * self.total_neuron_cores)
            # Expected tokens per core per second (this would need to be calibrated)
            expected_tokens_per_core = 500.0  # Placeholder value
            efficiency = (tokens_per_core / expected_tokens_per_core) * 100
            metrics["training_efficiency"] = round(min(efficiency, 100.0), 2)

        return metrics

    def calculate_metrics(self) -> dict[str, float]:
        """
        Calculate all enabled metrics for the current window.

        Returns:
            Dictionary of metric name -> value pairs to be added to logs
        """
        if self.window_start_time is None:
            return {}

        current_time = time.time()
        elapsed_time = current_time - self.window_start_time

        if elapsed_time <= 0:
            return {}

        metrics = {}

        # Throughput metrics (both general and per-core)
        metrics.update(self.calculate_throughput_metrics(elapsed_time))

        # MFU metrics
        metrics.update(self.calculate_mfu_metrics(elapsed_time))

        # Efficiency metrics
        metrics.update(self.calculate_efficiency_metrics(elapsed_time))

        # Add metadata for debugging/analysis
        if self.args.enable_throughput_metrics:
            metrics["metrics_window_steps"] = self.step_count
            metrics["metrics_window_duration"] = round(elapsed_time, 2)

        return metrics

    def should_calculate_metrics(self, step: int) -> bool:
        """
        Determine if metrics should be calculated at the current step.

        Args:
            step: Current training step

        Returns:
            True if metrics should be calculated and logged
        """
        if not any([self.args.enable_throughput_metrics,
                   self.args.enable_mfu_metrics,
                   self.args.enable_efficiency_metrics]):
            return False

        # Use metrics_logging_steps if specified, otherwise fall back to logging_steps
        metrics_logging_steps = getattr(self.args, 'metrics_logging_steps', None)
        if metrics_logging_steps is None:
            if hasattr(self.args, 'logging_steps') and self.args.logging_steps > 0:
                metrics_logging_steps = self.args.logging_steps
            else:
                return False

        return step > 0 and step % metrics_logging_steps == 0