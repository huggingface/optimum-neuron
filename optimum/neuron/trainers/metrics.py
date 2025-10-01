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
from typing import Any

import torch
import torch_xla.runtime as xr
from neuronx_distributed.parallel_layers.parallel_state import get_data_parallel_size

from ..trainers.training_args import NeuronTrainingArguments
from ..utils.training_utils import get_model_param_count


HARDWARE_TFLOPS = {
    # Ref: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium.html
    "trn1": 190 / 2,
    # Ref: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/trainium2.html
    "trn2": 667 / 2,
}


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

    def add_step(self, tokens: int, samples: int, step_time: float):
        """Add a new step to the moving window."""
        self.tokens_per_step.append(tokens)
        self.samples_per_step.append(samples)
        self.step_times.append(step_time)

    def get_window_stats(self) -> dict[str, float]:
        """Calculate statistics for the current window."""
        if not self.step_times:
            return {}

        total_tokens = sum(self.tokens_per_step)
        total_samples = sum(self.samples_per_step)
        total_time = sum(self.step_times)
        window_steps = len(self.step_times)

        return {
            "total_tokens": total_tokens,
            "total_samples": total_samples,
            "total_time": total_time,
            "window_steps": window_steps,
            "avg_tokens_per_step": total_tokens / window_steps if window_steps > 0 else 0,
            "avg_samples_per_step": total_samples / window_steps if window_steps > 0 else 0,
            "avg_time_per_step": total_time / window_steps if window_steps > 0 else 0,
        }

    def clear(self):
        """Clear the moving window."""
        self.tokens_per_step.clear()
        self.samples_per_step.clear()
        self.step_times.clear()

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
    - Individual metric timing control with start_metric()/stop_metric()
    - Moving average windows for stable real-time metrics
    - Comprehensive summary statistics for end-of-training analysis
    - Auto-detection of Trainium hardware for accurate MFU calculations
    - Dual metrics: train/ (real-time) and summary/ (end-of-training)
    - Configurable gradient accumulation cycle support for any metric

    Args:
        model: The model being trained
        training_args: NeuronTrainingArguments containing training configuration
        accumulating_metrics: List of metric names that should accumulate during gradient accumulation cycles.
                             Defaults to ["forward_pass", "backward_pass"]
    """

    def __init__(self, model: Any, training_args: NeuronTrainingArguments, accumulating_metrics: list[str] | None = None):
        self.model = model
        self.args = training_args

        # Configure which metrics should accumulate during gradient accumulation cycles
        self.accumulating_metrics = accumulating_metrics or ["forward_pass", "backward_pass"]

        # Input validation
        self._validate_inputs()

        # Hardware topology
        self.dp_size = get_data_parallel_size()
        self.total_neuron_cores = xr.world_size()

        # Model parameters (cached for MFU calculation)
        self.model_params = None
        if self.args.enable_mfu_metrics:
            # TODO: should we consider the full param count if case of TP / PP or just the local parameters?
            self.model_params = get_model_param_count(model, trainable_only=False)

        # Hardware specs (TFLOPS per core for bf16)
        self.peak_tflops_per_core = self._detect_hardware_tflops()

        # Moving average window configuration
        self.window_size = self.args.metrics_window_size

        # Per-metric moving windows and timing
        self.metric_windows = {}
        self.metric_start_times = {}
        self.current_batch_data = {}  # Store batch data for current metric timing

        # Cycle management for gradient accumulation
        self.cycle_active = False
        self.cycle_accumulators = dict.fromkeys(self.accumulating_metrics, 0.0)
        self.cycle_batch_data = {"tokens": 0, "samples": 0}
        self.component_start_time = None

        # Summary metrics collection (for end-of-training statistics)
        self.summary_metrics = {}  # Store all measurements for summary statistics

        self._initialize_metric_systems()

    def _validate_inputs(self):
        """Validate input parameters and configuration."""
        # Validate window size
        if not hasattr(self.args, "metrics_window_size"):
            raise ValueError("metrics_window_size not found in training arguments")

        if self.args.metrics_window_size <= 0:
            raise ValueError(f"metrics_window_size must be > 0, got {self.args.metrics_window_size}")

        # Validate model parameters for MFU if enabled
        if hasattr(self.args, "enable_mfu_metrics") and self.args.enable_mfu_metrics:
            if self.model is None:
                raise ValueError("Model cannot be None when MFU metrics are enabled")

        # Validate expected_tokens_per_core if provided
        if hasattr(self.args, "expected_tokens_per_core") and self.args.expected_tokens_per_core <= 0:
            raise ValueError(f"expected_tokens_per_core must be > 0, got {self.args.expected_tokens_per_core}")

    def _detect_hardware_tflops(self) -> float:
        """
        Auto-detect Trainium hardware type and return peak TFLOPS per core.

        Returns:
            Peak TFLOPS per core for bf16 operations
        """
        try:
            # Try to detect hardware from environment or other sources
            import os

            # Check environment variable first
            hardware_type = os.getenv("NEURON_HARDWARE_TYPE")
            if hardware_type and hardware_type in HARDWARE_TFLOPS:
                return HARDWARE_TFLOPS[hardware_type]

            # Try to detect from instance metadata or other sources
            # For now, default to trn1 but could be enhanced with actual detection logic
            instance_type = os.getenv("AWS_INSTANCE_TYPE", "")
            if "trn2" in instance_type.lower():
                return HARDWARE_TFLOPS["trn2"]
            elif "trn1" in instance_type.lower():
                return HARDWARE_TFLOPS["trn1"]

            # Default fallback to trn1
            return HARDWARE_TFLOPS["trn1"]

        except Exception:
            # If detection fails, fallback to trn1
            return HARDWARE_TFLOPS["trn1"]

    def _initialize_metric_systems(self):
        """Initialize per-metric moving windows and timing systems."""
        # Define available metrics
        metric_names = ["throughput", "mfu", "training_efficiency", "forward_pass", "backward_pass", "optimizer_step", "total_step"]

        # Initialize per-metric systems
        for metric_name in metric_names:
            self.metric_windows[metric_name] = MovingAverageWindow(self.window_size)
            self.metric_start_times[metric_name] = None
            self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}
            # Initialize summary metrics storage
            self.summary_metrics[metric_name] = {
                "step_times": [],
                "tokens_per_step": [],
                "samples_per_step": [],
                "step_numbers": [],
            }

    def start_gradient_accumulation_cycle(self):
        """
        Start a gradient accumulation cycle for cumulative timing.

        This method should be called at the beginning of each gradient accumulation cycle
        to enable proper cumulative timing for accumulating metrics.
        """
        self.cycle_active = True
        self.cycle_accumulators = dict.fromkeys(self.accumulating_metrics, 0.0)
        self.cycle_batch_data = {"tokens": 0, "samples": 0}

    def end_gradient_accumulation_cycle(self, step_number: int | None = None):
        """
        End a gradient accumulation cycle and record accumulated times.

        This method should be called at the end of each gradient accumulation cycle
        to record the cumulative times for all accumulating metrics.

        Args:
            step_number: Optional step number for tracking
        """
        if not self.cycle_active:
            return

        # Record accumulated times to windows for all accumulating metrics
        for metric_name in self.accumulating_metrics:
            if metric_name in self.cycle_accumulators:
                print(f"Accumulated time for {metric_name}: {self.cycle_accumulators[metric_name]:.4f} seconds")
                self.metric_windows[metric_name].add_step(
                    tokens=self.cycle_batch_data["tokens"],
                    samples=self.cycle_batch_data["samples"],
                    step_time=self.cycle_accumulators[metric_name],
                )

                # Add to summary metrics
                self.summary_metrics[metric_name]["step_times"].append(self.cycle_accumulators[metric_name])
                self.summary_metrics[metric_name]["tokens_per_step"].append(self.cycle_batch_data["tokens"])
                self.summary_metrics[metric_name]["samples_per_step"].append(self.cycle_batch_data["samples"])
                self.summary_metrics[metric_name]["step_numbers"].append(step_number or 0)

        # Reset cycle state
        self.cycle_active = False
        self.cycle_accumulators = dict.fromkeys(self.accumulating_metrics, 0.0)
        self.cycle_batch_data = {"tokens": 0, "samples": 0}

    def start_metric(self, metric_name: str, inputs: dict[str, Any] | None = None):
        """
        Start timing for a specific metric.

        Args:
            metric_name: Name of the metric to start ('throughput', 'mfu', 'training_efficiency', etc.)
            inputs: Optional batch inputs for token/sample counting
        """
        if metric_name not in self.metric_start_times:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(self.metric_start_times.keys())}")

        # Handle accumulation mode for accumulating metrics during gradient accumulation
        if self.cycle_active and metric_name in self.accumulating_metrics:
            # Start timing for this component within the accumulation cycle
            self.component_start_time = time.perf_counter()

            # Accumulate batch data for the cycle
            if inputs is not None:
                self._update_cycle_batch_data(inputs)
        else:
            # Normal single-step timing
            self.metric_start_times[metric_name] = time.perf_counter()

            # Reset batch data accumulator for this metric
            self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}

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

        self.current_batch_data[metric_name]["tokens"] += batch_tokens
        self.current_batch_data[metric_name]["samples"] += batch_samples

    def _update_cycle_batch_data(self, inputs: dict[str, Any]):
        """Helper method to count tokens and samples for cycle accumulation."""
        batch_tokens = 0
        batch_samples = 0

        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            if isinstance(input_ids, torch.Tensor):
                batch_tokens = input_ids.numel()
                batch_samples = input_ids.size(0)

        self.cycle_batch_data["tokens"] += batch_tokens
        self.cycle_batch_data["samples"] += batch_samples

    def stop_metric(self, metric_name: str, step_number: int | None = None):
        """
        Stop timing for a specific metric and add the measurement to its moving window.

        Args:
            metric_name: Name of the metric to stop
            step_number: Optional step number for tracking
        """
        if metric_name not in self.metric_start_times:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(self.metric_start_times.keys())}")

        # Handle accumulation mode for accumulating metrics during gradient accumulation
        if self.cycle_active and metric_name in self.accumulating_metrics:
            if self.component_start_time is None:
                # Component wasn't started, ignore
                return

            # Calculate elapsed time for this component
            elapsed_time = time.perf_counter() - self.component_start_time

            # Accumulate time for this component
            self.cycle_accumulators[metric_name] += elapsed_time

            # Reset component timing
            self.component_start_time = None
        else:
            # Normal single-step timing
            if self.metric_start_times[metric_name] is None:
                # Metric wasn't started, ignore
                return

            # Calculate elapsed time
            elapsed_time = time.perf_counter() - self.metric_start_times[metric_name]

            # Get batch data for this metric
            batch_data = self.current_batch_data[metric_name]

            # Add to moving window
            self.metric_windows[metric_name].add_step(
                tokens=batch_data["tokens"],
                samples=batch_data["samples"],
                step_time=elapsed_time,
            )

            # Add to summary metrics for end-of-training statistics
            self.summary_metrics[metric_name]["step_times"].append(elapsed_time)
            self.summary_metrics[metric_name]["tokens_per_step"].append(batch_data["tokens"])
            self.summary_metrics[metric_name]["samples_per_step"].append(batch_data["samples"])
            self.summary_metrics[metric_name]["step_numbers"].append(step_number or 0)

            # Reset the metric state
            self.metric_start_times[metric_name] = None
            self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}

    def calculate_metric(self, metric_name: str) -> dict[str, float]:
        """
        Calculate a specific metric using its moving window data.

        Args:
            metric_name: Name of the metric to calculate ('throughput', 'mfu', 'training_efficiency', 'all')

        Returns:
            Dictionary containing the calculated metrics for the specified type.
        """
        metrics = {}

        throughput_metrics = {}
        if metric_name in ["throughput", "all"] and self.args.enable_throughput_metrics:
            throughput_metrics = self._calculate_throughput_metrics_from_window("throughput")
            if metric_name in ["throughput", "all"]:
                metrics.update(throughput_metrics)

        if metric_name in ["mfu", "all"]:
            if self.args.enable_mfu_metrics:
                metrics.update(self._calculate_mfu_metrics_from_window("mfu"))

        if metric_name in ["training_efficiency", "all"]:
            if self.args.enable_efficiency_metrics:
                metrics.update(self._calculate_training_efficiency_metrics())

        return metrics

    def calculate_summary_metrics(self) -> dict[str, float]:
        """
        Calculate comprehensive summary metrics across all training steps.

        Returns:
            Dictionary containing summary statistics (avg, min, max, etc.) for all metrics.
        """
        summary = {}

        # Calculate throughput summary if enabled
        if self.args.enable_throughput_metrics and self.summary_metrics["throughput"]["step_times"]:
            summary.update(self._calculate_throughput_summary())

        # Calculate MFU summary if enabled
        if self.args.enable_mfu_metrics and self.summary_metrics["mfu"]["step_times"]:
            summary.update(self._calculate_mfu_summary())


        # Calculate training efficiency summary if enabled and component data is available
        if self.args.enable_efficiency_metrics:
            training_efficiency_summary = self._calculate_training_efficiency_summary()
            if training_efficiency_summary:
                summary.update(training_efficiency_summary)

        return summary

    def _calculate_throughput_summary(self) -> dict[str, float]:
        """Calculate summary statistics for throughput metrics."""
        summary = {}
        metric_data = self.summary_metrics["throughput"]

        if not metric_data["step_times"]:
            return summary

        step_times = metric_data["step_times"]
        tokens_per_step = metric_data["tokens_per_step"]
        samples_per_step = metric_data["samples_per_step"]

        # Calculate per-step throughput rates
        local_tokens_per_sec_values = [
            tokens / time if time > 0 else 0 for tokens, time in zip(tokens_per_step, step_times)
        ]
        local_samples_per_sec_values = [
            samples / time if time > 0 else 0 for samples, time in zip(samples_per_step, step_times)
        ]

        # Global effective throughput (accounting for data parallelism)
        global_tokens_per_sec_values = [rate * self.dp_size for rate in local_tokens_per_sec_values]
        global_samples_per_sec_values = [rate * self.dp_size for rate in local_samples_per_sec_values]

        # Per-core metrics (for hardware utilization)
        tokens_per_sec_per_core_values = [rate / self.total_neuron_cores for rate in local_tokens_per_sec_values]
        samples_per_sec_per_core_values = [rate / self.total_neuron_cores for rate in local_samples_per_sec_values]

        # Summary statistics for throughput
        if global_tokens_per_sec_values:
            summary.update(
                {
                    "summary/tokens_per_sec_avg": sum(global_tokens_per_sec_values)
                    / len(global_tokens_per_sec_values),
                    "summary/tokens_per_sec_min": min(global_tokens_per_sec_values),
                    "summary/tokens_per_sec_max": max(global_tokens_per_sec_values),
                    "summary/tokens_per_sec_per_core_avg": sum(tokens_per_sec_per_core_values)
                    / len(tokens_per_sec_per_core_values),
                    "summary/tokens_per_sec_per_core_min": min(tokens_per_sec_per_core_values),
                    "summary/tokens_per_sec_per_core_max": max(tokens_per_sec_per_core_values),
                }
            )

        if global_samples_per_sec_values:
            summary.update(
                {
                    "summary/samples_per_sec_avg": sum(global_samples_per_sec_values)
                    / len(global_samples_per_sec_values),
                    "summary/samples_per_sec_min": min(global_samples_per_sec_values),
                    "summary/samples_per_sec_max": max(global_samples_per_sec_values),
                    "summary/samples_per_sec_per_core_avg": sum(samples_per_sec_per_core_values)
                    / len(samples_per_sec_per_core_values),
                    "summary/samples_per_sec_per_core_min": min(samples_per_sec_per_core_values),
                    "summary/samples_per_sec_per_core_max": max(samples_per_sec_per_core_values),
                }
            )

        # Step timing statistics
        summary.update(
            {
                "summary/step_time_avg": sum(step_times) / len(step_times),
                "summary/step_time_min": min(step_times),
                "summary/step_time_max": max(step_times),
                "summary/total_training_steps": len(step_times),
                "summary/total_tokens_processed": sum(tokens_per_step),
                "summary/total_samples_processed": sum(samples_per_step),
            }
        )

        return summary

    def _calculate_mfu_summary(self) -> dict[str, float]:
        """Calculate summary statistics for MFU metrics."""
        summary = {}
        metric_data = self.summary_metrics["mfu"]

        if not metric_data["step_times"] or self.model_params is None:
            return summary

        step_times = metric_data["step_times"]
        tokens_per_step = metric_data["tokens_per_step"]

        # Calculate MFU for each step
        mfu_values = []
        for tokens, t in zip(tokens_per_step, step_times):
            if t > 0 and tokens > 0:
                theoretical_flops = 18 * self.model_params * tokens
                actual_flops_per_sec = theoretical_flops / t
                peak_flops_per_sec = self.peak_tflops_per_core * 1e12 * self.total_neuron_cores
                mfu_percentage = (actual_flops_per_sec / peak_flops_per_sec) * 100
                mfu_values.append(mfu_percentage)

        if mfu_values:
            summary.update(
                {
                    "summary/mfu_avg": sum(mfu_values) / len(mfu_values),
                    "summary/mfu_min": min(mfu_values),
                    "summary/mfu_max": max(mfu_values),
                }
            )

        return summary


    def _calculate_training_efficiency_summary(self) -> dict[str, float]:
        """Calculate summary statistics for training efficiency metrics."""
        summary = {}

        # Get component metric data
        forward_data = self.summary_metrics.get("forward_pass", {})
        backward_data = self.summary_metrics.get("backward_pass", {})
        optimizer_data = self.summary_metrics.get("optimizer_step", {})
        total_data = self.summary_metrics.get("total_step", {})

        # Check if we have data for all components
        if not all([
            forward_data.get("step_times", []),
            backward_data.get("step_times", []),
            optimizer_data.get("step_times", []),
            total_data.get("step_times", [])
        ]):
            return summary

        # Calculate training efficiency for each step
        efficiency_values = []
        overhead_values = []

        forward_times = forward_data["step_times"]
        backward_times = backward_data["step_times"]
        optimizer_times = optimizer_data["step_times"]
        total_times = total_data["step_times"]

        # Ensure all lists have the same length (take minimum)
        min_steps = min(len(forward_times), len(backward_times), len(optimizer_times), len(total_times))

        for i in range(min_steps):
            forward_time = forward_times[i]
            backward_time = backward_times[i]
            optimizer_time = optimizer_times[i]
            total_time = total_times[i]

            if total_time > 0:
                compute_time = forward_time + backward_time + optimizer_time
                overhead_time = total_time - compute_time

                efficiency = (compute_time / total_time) * 100
                overhead = (overhead_time / total_time) * 100

                efficiency_values.append(efficiency)
                overhead_values.append(overhead)

        if efficiency_values:
            summary.update({
                "summary/training_efficiency_avg": round(sum(efficiency_values) / len(efficiency_values), 2),
                "summary/training_efficiency_min": round(min(efficiency_values), 2),
                "summary/training_efficiency_max": round(max(efficiency_values), 2),
                "summary/training_overhead_avg": round(sum(overhead_values) / len(overhead_values), 2),
                "summary/training_overhead_min": round(min(overhead_values), 2),
                "summary/training_overhead_max": round(max(overhead_values), 2),
            })

        return summary

    def _calculate_throughput_metrics_from_window(self, metric_name: str) -> dict[str, float]:
        """Calculate throughput metrics from a specific metric window."""
        if metric_name not in self.metric_windows or self.metric_windows[metric_name].size == 0:
            return {}

        window_stats = self.metric_windows[metric_name].get_window_stats()
        if not window_stats or window_stats["total_time"] <= 0:
            return {}

        total_tokens = window_stats["total_tokens"]
        total_samples = window_stats["total_samples"]
        total_time = window_stats["total_time"]

        metrics = {}

        # General throughput metrics (for training performance comparison)
        # These account for data parallelism to show effective global throughput
        if total_tokens > 0:
            local_tokens_per_sec = total_tokens / total_time
            metrics["train/tokens_per_sec"] = local_tokens_per_sec * self.dp_size  # Global effective throughput
            metrics["train/avg_tokens_per_step"] = window_stats["avg_tokens_per_step"]

        if total_samples > 0:
            local_samples_per_sec = total_samples / total_time
            metrics["train/samples_per_sec"] = local_samples_per_sec * self.dp_size  # Global effective throughput
            metrics["train/avg_samples_per_step"] = window_stats["avg_samples_per_step"]

        # Per-neuron-core metrics (for hardware utilization analysis)
        if self.total_neuron_cores > 0:
            if total_tokens > 0:
                metrics["train/tokens_per_sec_per_neuron_core"] = total_tokens / (total_time * self.total_neuron_cores)
            if total_samples > 0:
                metrics["train/samples_per_sec_per_neuron_core"] = total_samples / (
                    total_time * self.total_neuron_cores
                )

        # Additional window information
        metrics["train/avg_step_time"] = window_stats["avg_time_per_step"]

        return metrics

    def _calculate_mfu_metrics_from_window(self, metric_name: str) -> dict[str, float]:
        """Calculate MFU metrics from a specific metric window."""
        if (
            metric_name not in self.metric_windows
            or self.model_params is None
            or self.metric_windows[metric_name].size == 0
        ):
            return {}

        window_stats = self.metric_windows[metric_name].get_window_stats()
        if not window_stats or window_stats["total_time"] <= 0 or window_stats["total_tokens"] == 0:
            return {}

        total_tokens = window_stats["total_tokens"]
        total_time = window_stats["total_time"]

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
            "train/mfu": round(mfu_percentage, 2),
        }

    def _get_metric_average_time(self, metric_name: str) -> float:
        """
        Get average step time for a metric from its window.

        Args:
            metric_name: Name of the metric to get timing data for

        Returns:
            Average time per step for the metric, or 0.0 if metric doesn't exist
        """
        if metric_name not in self.metric_windows:
            return 0.0
        window_stats = self.metric_windows[metric_name].get_window_stats()
        return window_stats.get("avg_time_per_step", 0.0)

    def _calculate_training_efficiency_metrics(self) -> dict[str, float]:
        """
        Calculate training efficiency from component timing metrics.

        Training efficiency = (forward + backward + optimizer time) / total time
        This measures how much of the total training time is spent on useful computation
        vs overhead (data loading, synchronization, etc.).

        Returns:
            Dictionary containing training efficiency percentage
        """
        # Get component times from individual metric windows
        forward_time = self._get_metric_average_time("forward_pass")
        backward_time = self._get_metric_average_time("backward_pass")
        optimizer_time = self._get_metric_average_time("optimizer_step")
        total_time = self._get_metric_average_time("total_step")

        if total_time <= 0:
            return {}

        # Calculate useful compute time vs total time
        compute_time = forward_time + backward_time + optimizer_time
        overhead_time = total_time - compute_time

        efficiency_percentage = (compute_time / total_time) * 100
        overhead_percentage = (overhead_time / total_time) * 100

        return {
            "train/training_efficiency": round(efficiency_percentage, 2),
            "train/training_overhead": round(overhead_percentage, 2),
        }

    def reset_window(self):
        """Reset moving average windows and timing (but preserve summary metrics)."""
        for metric_name in self.metric_windows:
            self.metric_windows[metric_name].clear()
            self.metric_start_times[metric_name] = None
            self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}

    def reset_all_metrics(self):
        """Reset all metrics including summary metrics (for training restart)."""
        self.reset_window()
        for metric_name in self.summary_metrics:
            self.summary_metrics[metric_name] = {
                "step_times": [],
                "tokens_per_step": [],
                "samples_per_step": [],
                "step_numbers": [],
            }

    def should_calculate_metrics(self, step: int) -> bool:
        """
        Determine if metrics should be calculated at the current step.

        Args:
            step: Current training step

        Returns:
            True if metrics should be calculated and logged
        """
        if not any(
            [self.args.enable_throughput_metrics, self.args.enable_mfu_metrics, self.args.enable_efficiency_metrics]
        ):
            return False

        # Use metrics_logging_steps if specified, otherwise fall back to logging_steps
        metrics_logging_steps = self.args.metrics_logging_steps
        if metrics_logging_steps is None:
            metrics_logging_steps = self.args.logging_steps
        if metrics_logging_steps > 0:
            return step > 0 and step % metrics_logging_steps == 0
        return False
