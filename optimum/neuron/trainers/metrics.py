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
from collections import deque
from typing import Any, Dict, Optional

import torch
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_size,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)

from ..utils.training_utils import get_model_param_count


class MetricsClock:
    """
    A clock for measuring elapsed time with different timing mechanisms.

    Supports multiple timing methods:
    - 'wall_time': Standard wall clock time
    - 'process_time': Process CPU time
    - 'perf_counter': High-resolution performance counter
    """

    def __init__(self, clock_type: str = 'wall_time'):
        self.clock_type = clock_type
        self.start_time = None

        if clock_type == 'wall_time':
            self.time_func = time.time
        elif clock_type == 'process_time':
            self.time_func = time.process_time
        elif clock_type == 'perf_counter':
            self.time_func = time.perf_counter
        else:
            raise ValueError(f"Unsupported clock type: {clock_type}")

    def start(self):
        """Start the clock."""
        self.start_time = self.time_func()

    def elapsed(self) -> Optional[float]:
        """Get elapsed time since start. Returns None if not started."""
        if self.start_time is None:
            return None
        return self.time_func() - self.start_time

    def reset(self):
        """Reset the clock."""
        self.start_time = None


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
        self.step_numbers = deque(maxlen=window_size)

    def add_step(self, tokens: int, samples: int, step_time: float, step_number: int):
        """Add a new step to the moving window."""
        self.tokens_per_step.append(tokens)
        self.samples_per_step.append(samples)
        self.step_times.append(step_time)
        self.step_numbers.append(step_number)

    def get_window_stats(self) -> Dict[str, float]:
        """Calculate statistics for the current window."""
        if not self.step_times:
            return {}

        total_tokens = sum(self.tokens_per_step)
        total_samples = sum(self.samples_per_step)
        total_time = sum(self.step_times)
        window_steps = len(self.step_times)

        return {
            'total_tokens': total_tokens,
            'total_samples': total_samples,
            'total_time': total_time,
            'window_steps': window_steps,
            'avg_tokens_per_step': total_tokens / window_steps if window_steps > 0 else 0,
            'avg_samples_per_step': total_samples / window_steps if window_steps > 0 else 0,
            'avg_time_per_step': total_time / window_steps if window_steps > 0 else 0,
        }

    def clear(self):
        """Clear the moving window."""
        self.tokens_per_step.clear()
        self.samples_per_step.clear()
        self.step_times.clear()
        self.step_numbers.clear()

    @property
    def is_full(self) -> bool:
        """Check if the window is full."""
        return len(self.step_times) == self.window_size

    @property
    def size(self) -> int:
        """Get current window size."""
        return len(self.step_times)


class TrainingMetricsCollector:
    """
    Collects and calculates training performance metrics for Neuron distributed training.

    Provides both per-neuron-core metrics (for hardware utilization analysis) and
    general throughput metrics (for training performance comparison).

    Features:
    - Individual metric clock control with start_metric()/stop_metric()
    - Moving average windows for stable metrics
    - Multiple clocks for different timing measurements
    - Configurable window sizes and clock types
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
            # TODO: should we consider the full param count if case of TP / PP or just the local parameters?
            self.model_params = get_model_param_count(model, trainable_only=False)

        # Hardware specs (TFLOPS per core for bf16)
        self.peak_tflops_per_core = getattr(self.args, 'peak_tflops_per_core', 100.0)

        # Moving average window configuration
        self.window_size = getattr(self.args, 'metrics_window_size', 50)  # Default 50 steps

        # Per-metric moving windows and clocks
        self.metric_windows = {}
        self.metric_clocks = {}
        self.metric_start_times = {}
        self.current_batch_data = {}  # Store batch data for current metric timing

        self._initialize_metric_systems()

    def _initialize_metric_systems(self):
        """Initialize per-metric moving windows and clocks."""
        # Define available metrics with their default clock types
        metric_configs = {
            'throughput': 'perf_counter',  # High-resolution for throughput
            'mfu': 'wall_time',            # Wall time for MFU calculations
            'efficiency': 'process_time',  # Process time for efficiency
        }

        # Add custom metrics from config
        custom_clocks = getattr(self.args, 'metrics_clocks', {})
        metric_configs.update(custom_clocks)

        # Initialize per-metric systems
        for metric_name, clock_type in metric_configs.items():
            self.metric_windows[metric_name] = MovingAverageWindow(self.window_size)
            self.metric_clocks[metric_name] = MetricsClock(clock_type)
            self.metric_start_times[metric_name] = None
            self.current_batch_data[metric_name] = {'tokens': 0, 'samples': 0}

    def start_metric(self, metric_name: str, inputs: dict[str, Any] = None):
        """
        Start timing for a specific metric.

        Args:
            metric_name: Name of the metric to start ('throughput', 'mfu', 'efficiency', etc.)
            inputs: Optional batch inputs for token/sample counting
        """
        if metric_name not in self.metric_clocks:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(self.metric_clocks.keys())}")

        # Start the clock for this metric
        self.metric_clocks[metric_name].start()
        self.metric_start_times[metric_name] = self.metric_clocks[metric_name].time_func()

        # Reset batch data accumulator for this metric
        self.current_batch_data[metric_name] = {'tokens': 0, 'samples': 0}

        # Count tokens and samples if inputs provided
        if inputs is not None:
            self._update_batch_data(metric_name, inputs)

    def update_metric_batch_data(self, metric_name: str, inputs: dict[str, Any]):
        """
        Update batch data for a specific metric (for accumulation across gradient accumulation steps).

        Args:
            metric_name: Name of the metric to update
            inputs: Batch inputs containing 'input_ids' for token counting
        """
        if metric_name not in self.current_batch_data:
            return

        self._update_batch_data(metric_name, inputs)

    def _update_batch_data(self, metric_name: str, inputs: dict[str, Any]):
        """Helper method to count tokens and samples from inputs."""
        batch_tokens = 0
        batch_samples = 0

        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                batch_tokens = input_ids.numel()
                batch_samples = input_ids.size(0)

        self.current_batch_data[metric_name]['tokens'] += batch_tokens
        self.current_batch_data[metric_name]['samples'] += batch_samples

    def stop_metric(self, metric_name: str, step_number: int = None):
        """
        Stop timing for a specific metric and add the measurement to its moving window.

        Args:
            metric_name: Name of the metric to stop
            step_number: Optional step number for tracking
        """
        if metric_name not in self.metric_clocks:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(self.metric_clocks.keys())}")

        if self.metric_start_times[metric_name] is None:
            # Metric wasn't started, ignore
            return

        # Calculate elapsed time
        current_time = self.metric_clocks[metric_name].time_func()
        elapsed_time = current_time - self.metric_start_times[metric_name]

        # Get batch data for this metric
        batch_data = self.current_batch_data[metric_name]

        # Add to moving window
        self.metric_windows[metric_name].add_step(
            tokens=batch_data['tokens'],
            samples=batch_data['samples'],
            step_time=elapsed_time,
            step_number=step_number or 0
        )

        # Reset the metric state
        self.metric_start_times[metric_name] = None
        self.current_batch_data[metric_name] = {'tokens': 0, 'samples': 0}

    def finalize_metric(self, metric_name: str) -> dict[str, float]:
        """
        Calculate a specific metric using its moving window data.

        Args:
            metric_name: Name of the metric to calculate ('throughput', 'mfu', 'efficiency', 'all')

        Returns:
            Dictionary containing the calculated metrics for the specified type.
        """
        if metric_name == 'throughput':
            if self.args.enable_throughput_metrics:
                return self._calculate_throughput_metrics_from_window('throughput')
        elif metric_name == 'mfu':
            if self.args.enable_mfu_metrics:
                return self._calculate_mfu_metrics_from_window('mfu')
        elif metric_name == 'efficiency':
            if self.args.enable_efficiency_metrics:
                return self._calculate_efficiency_metrics_from_window('efficiency')
        elif metric_name == 'all':
            return self.calculate_all_metrics()
        else:
            # Check if it's a custom metric name
            if metric_name in self.metric_windows:
                # Default to throughput calculation for custom metrics
                return self._calculate_throughput_metrics_from_window(metric_name)

            raise ValueError(
                f"Unknown metric: '{metric_name}'. "
                f"Available metrics: {list(self.metric_windows.keys())} or 'all'"
            )

        return {}

    def _calculate_throughput_metrics_from_window(self, metric_name: str) -> dict[str, float]:
        """Calculate throughput metrics from a specific metric window."""
        if metric_name not in self.metric_windows or self.metric_windows[metric_name].size == 0:
            return {}

        window_stats = self.metric_windows[metric_name].get_window_stats()
        if not window_stats or window_stats['total_time'] <= 0:
            return {}

        total_tokens = window_stats['total_tokens']
        total_samples = window_stats['total_samples']
        total_time = window_stats['total_time']

        metrics = {}

        # General throughput metrics (for training performance comparison)
        if total_tokens > 0:
            metrics["tokens_per_sec"] = total_tokens / total_time
            metrics["avg_tokens_per_step"] = window_stats['avg_tokens_per_step']

        if total_samples > 0:
            metrics["samples_per_sec"] = total_samples / total_time
            metrics["avg_samples_per_step"] = window_stats['avg_samples_per_step']

        # Per-neuron-core metrics (for hardware utilization analysis)
        if self.total_neuron_cores > 0:
            if total_tokens > 0:
                metrics["tokens_per_sec_per_neuron_core"] = total_tokens / (total_time * self.total_neuron_cores)
            if total_samples > 0:
                metrics["samples_per_sec_per_neuron_core"] = total_samples / (total_time * self.total_neuron_cores)

        # Additional window information
        metrics["metrics_window_steps"] = window_stats['window_steps']
        metrics["avg_step_time"] = window_stats['avg_time_per_step']
        metrics["metrics_window_size"] = self.window_size
        metrics["window_is_full"] = self.metric_windows[metric_name].is_full

        return metrics

    def _calculate_mfu_metrics_from_window(self, metric_name: str) -> dict[str, float]:
        """Calculate MFU metrics from a specific metric window."""
        if (metric_name not in self.metric_windows or
            self.model_params is None or
            self.metric_windows[metric_name].size == 0):
            return {}

        window_stats = self.metric_windows[metric_name].get_window_stats()
        if not window_stats or window_stats['total_time'] <= 0 or window_stats['total_tokens'] == 0:
            return {}

        total_tokens = window_stats['total_tokens']
        total_time = window_stats['total_time']

        # Theoretical FLOPs calculation for transformers
        # Forward pass: ~6 * params * tokens (rough approximation)
        # Backward pass: ~2 * forward pass FLOPs
        # Total: ~18 * params * tokens
        theoretical_flops = 18 * self.model_params * total_tokens

        # Actual FLOPs per second achieved
        actual_flops_per_sec = theoretical_flops / total_time

        # Peak FLOPs available across all cores
        peak_flops_per_sec = self.peak_tflops_per_core * 1e12 * self.total_neuron_cores

        # MFU as percentage
        mfu_percentage = (actual_flops_per_sec / peak_flops_per_sec) * 100

        return {
            "model_flops_utilization": round(mfu_percentage, 2),
            "theoretical_flops_per_sec": actual_flops_per_sec,
            "peak_flops_per_sec": peak_flops_per_sec,
        }

    def _calculate_efficiency_metrics_from_window(self, metric_name: str) -> dict[str, float]:
        """Calculate efficiency metrics from a specific metric window."""
        if metric_name not in self.metric_windows or self.metric_windows[metric_name].size == 0:
            return {}

        # Get throughput metrics first
        throughput_metrics = self._calculate_throughput_metrics_from_window(metric_name)
        if not throughput_metrics:
            return {}

        metrics = {}

        # Simple efficiency based on tokens per second per core vs expected
        if "tokens_per_sec_per_neuron_core" in throughput_metrics:
            tokens_per_core = throughput_metrics["tokens_per_sec_per_neuron_core"]
            expected_tokens_per_core = getattr(self.args, 'expected_tokens_per_core', 500.0)
            efficiency = (tokens_per_core / expected_tokens_per_core) * 100
            metrics["training_efficiency"] = round(min(efficiency, 100.0), 2)

        # Consistency metric: coefficient of variation of step times
        window_stats = self.metric_windows[metric_name].get_window_stats()
        if window_stats['window_steps'] > 1:
            step_times = list(self.metric_windows[metric_name].step_times)
            if len(step_times) > 1:
                mean_time = sum(step_times) / len(step_times)
                variance = sum((t - mean_time) ** 2 for t in step_times) / len(step_times)
                std_dev = variance ** 0.5
                cv = (std_dev / mean_time) * 100 if mean_time > 0 else 0
                metrics["step_time_consistency"] = round(100 - min(cv, 100), 2)  # Higher is better

        return metrics

    def calculate_all_metrics(self) -> dict[str, float]:
        """Calculate all enabled metrics and combine them."""
        metrics = {}

        if self.args.enable_throughput_metrics:
            metrics.update(self._calculate_throughput_metrics_from_window('throughput'))

        if self.args.enable_mfu_metrics:
            metrics.update(self._calculate_mfu_metrics_from_window('mfu'))

        if self.args.enable_efficiency_metrics:
            metrics.update(self._calculate_efficiency_metrics_from_window('efficiency'))

        return metrics

    def reset_window(self):
        """Reset all metric windows and clocks."""
        for metric_name in self.metric_windows:
            self.metric_windows[metric_name].clear()
            self.metric_clocks[metric_name].reset()
            self.metric_start_times[metric_name] = None
            self.current_batch_data[metric_name] = {'tokens': 0, 'samples': 0}

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
