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
from contextlib import contextmanager
from typing import Any

import torch
import torch_xla.runtime as xr
from torch_neuronx.utils import get_platform_target

from ...utils.training_utils import get_model_param_count
from ..training_args import NeuronTrainingArguments
from .base import MetricPlugin
from .constants import HARDWARE_TFLOPS
from .efficiency import EfficiencyPlugin
from .mfu import MFUPlugin
from .registry import PluginRegistry
from .throughput import ThroughputPlugin
from .timing import ComponentTimingPlugin
from .window import MovingAverageWindow


class TrainingMetricsCollector:
    """
    Tracks training performance metrics using a plugin system.

    Provides real-time metrics during training and summary stats at the end.
    Auto-detects Trainium hardware and handles distributed training setups.
    """

    def __init__(
        self,
        model: Any,
        training_args: NeuronTrainingArguments,
        custom_plugins: list[MetricPlugin] | None = None,
    ):
        self.model = model
        self.args = training_args

        # Set up plugins
        all_plugins = self._get_default_plugins() + (custom_plugins or [])
        self.registry = PluginRegistry(all_plugins)
        self.registry.validate_dependencies()

        # Check if any metrics are enabled
        self.enabled = any(plugin.is_enabled(training_args) for plugin in all_plugins)

        if not self.enabled:
            self.metric_windows = {}
            self.metric_start_times = {}
            self.current_batch_data = {}
            self.summary_metrics = {}
            return

        self._validate_inputs()

        self.dp_size = self.args.trn_config.data_parallel_size
        self.total_neuron_cores = xr.world_size()

        self.model_params = None
        if self.args.enable_mfu_metrics:
            self.model_params = get_model_param_count(model, trainable_only=False)

        self.peak_tflops_per_core = self._detect_hardware_tflops()
        self.window_size = self.args.metrics_window_size

        # Only work with enabled plugins
        self.active_plugins = [p for p in all_plugins if p.is_enabled(training_args)]

        # Initialize metric windows and tracking for active plugins
        self.metric_windows = {}
        self.metric_start_times = {}
        self.current_batch_data = {}
        self.summary_metrics = {}
        self.accumulating_metrics = set()

        for plugin in self.active_plugins:
            for metric_name in plugin.get_metric_names():
                self.metric_windows[metric_name] = MovingAverageWindow(self.window_size)
                self.metric_start_times[metric_name] = None
                self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}
                self.summary_metrics[metric_name] = {
                    "step_times": [],
                    "tokens_per_step": [],
                    "samples_per_step": [],
                    "step_numbers": [],
                }
                if plugin.requires_accumulation:
                    self.accumulating_metrics.add(metric_name)

        self.cycle_active = False
        self.cycle_accumulators = dict.fromkeys(self.accumulating_metrics, 0.0)
        self.cycle_batch_data = {"tokens": 0, "samples": 0}
        self.component_start_time = None

    def _get_default_plugins(self) -> list[MetricPlugin]:
        return [
            ThroughputPlugin(),
            MFUPlugin(),
            EfficiencyPlugin(),
            ComponentTimingPlugin(),
        ]

    def _validate_inputs(self):
        if self.args.metrics_window_size <= 0:
            raise ValueError(f"metrics_window_size must be > 0, got {self.args.metrics_window_size}")

        if self.args.enable_mfu_metrics and self.model is None:
            raise ValueError("Model cannot be None when MFU metrics are enabled")

    def _detect_hardware_tflops(self) -> float:
        platform_target = get_platform_target().lower()
        if platform_target not in HARDWARE_TFLOPS:
            raise ValueError(f"Unknown platform '{platform_target}'. We support: {list(HARDWARE_TFLOPS.keys())}")
        return HARDWARE_TFLOPS[platform_target]

    def _should_calculate_plugin(self, plugin: MetricPlugin, metric_type: str) -> bool:
        if metric_type == "all":
            return True
        if plugin.name == metric_type:
            return True
        if hasattr(plugin, "handles_metric") and plugin.handles_metric(metric_type):
            return True
        return False

    def _get_plugin_window_stats(self, plugin: MetricPlugin) -> dict:
        if hasattr(plugin, "get_metric_names") and len(plugin.get_metric_names()) > 1:
            # Multi-metric plugins use inter-plugin communication instead
            return {}
        else:
            # Single-metric plugins get their own window stats
            metric_name = plugin.name
            if metric_name in self.metric_windows:
                return self.metric_windows[metric_name].get_window_stats()
            return {}

    def get_metric_average_time(self, metric_name: str) -> float:
        if metric_name not in self.metric_windows:
            return 0.0
        window_stats = self.metric_windows[metric_name].get_window_stats()
        return window_stats.get("avg_time_per_step", 0.0)

    def get_metric_window_stats(self, metric_name: str) -> dict:
        if metric_name not in self.metric_windows:
            return {}
        return self.metric_windows[metric_name].get_window_stats()

    def start_gradient_accumulation_cycle(self):
        """Start accumulating timing across multiple forward/backward passes."""
        if not self.enabled:
            return
        self.cycle_active = True
        self.cycle_accumulators = dict.fromkeys(self.accumulating_metrics, 0.0)
        self.cycle_batch_data = {"tokens": 0, "samples": 0}

    def end_gradient_accumulation_cycle(self, step_number: int | None = None):
        """Finish accumulation cycle and record the total times."""
        if not self.enabled or not self.cycle_active:
            return

        for metric_name in self.accumulating_metrics:
            if metric_name in self.cycle_accumulators:
                self.metric_windows[metric_name].add_step(
                    tokens=self.cycle_batch_data["tokens"],
                    samples=self.cycle_batch_data["samples"],
                    step_time=self.cycle_accumulators[metric_name],
                )

                self.summary_metrics[metric_name]["step_times"].append(self.cycle_accumulators[metric_name])
                self.summary_metrics[metric_name]["tokens_per_step"].append(self.cycle_batch_data["tokens"])
                self.summary_metrics[metric_name]["samples_per_step"].append(self.cycle_batch_data["samples"])
                self.summary_metrics[metric_name]["step_numbers"].append(step_number or 0)

        self.cycle_active = False
        self.cycle_accumulators = dict.fromkeys(self.accumulating_metrics, 0.0)
        self.cycle_batch_data = {"tokens": 0, "samples": 0}

    def start_metric(self, metric_name: str, inputs: dict[str, Any] | None = None):
        """Start timing a metric."""
        if not self.enabled:
            return
        if metric_name not in self.metric_start_times:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(self.metric_start_times.keys())}")

        if self.cycle_active and metric_name in self.accumulating_metrics:
            self.component_start_time = time.perf_counter()
            if inputs is not None:
                self._update_cycle_batch_data(inputs)
        else:
            self.metric_start_times[metric_name] = time.perf_counter()
            self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}
            if inputs is not None:
                self._update_batch_data(metric_name, inputs)

    def update_metric_batch_data(self, metric_name: str, inputs: dict[str, Any]):
        if not self.enabled or metric_name not in self.current_batch_data:
            return
        self._update_batch_data(metric_name, inputs)

    @contextmanager
    def time_metric(self, metric_name: str, inputs: dict[str, Any] | None = None, step_number: int | None = None):
        """Context manager for timing - handles start/stop automatically."""
        if not self.enabled:
            yield
            return

        self.start_metric(metric_name, inputs)
        try:
            yield
        finally:
            self.stop_metric(metric_name, step_number)

    def _update_batch_data(self, metric_name: str, inputs: dict[str, Any]):
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
        """Stop timing and record the measurement."""
        if not self.enabled:
            return
        if metric_name not in self.metric_start_times:
            raise ValueError(f"Unknown metric: {metric_name}. Available: {list(self.metric_start_times.keys())}")

        if self.cycle_active and metric_name in self.accumulating_metrics:
            if self.component_start_time is None:
                return

            elapsed_time = time.perf_counter() - self.component_start_time
            self.cycle_accumulators[metric_name] += elapsed_time
            self.component_start_time = None
        else:
            if self.metric_start_times[metric_name] is None:
                return

            elapsed_time = time.perf_counter() - self.metric_start_times[metric_name]
            batch_data = self.current_batch_data[metric_name]

            self.metric_windows[metric_name].add_step(
                tokens=batch_data["tokens"],
                samples=batch_data["samples"],
                step_time=elapsed_time,
            )

            self.summary_metrics[metric_name]["step_times"].append(elapsed_time)
            self.summary_metrics[metric_name]["tokens_per_step"].append(batch_data["tokens"])
            self.summary_metrics[metric_name]["samples_per_step"].append(batch_data["samples"])
            self.summary_metrics[metric_name]["step_numbers"].append(step_number or 0)

            self.metric_start_times[metric_name] = None
            self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}

    def calculate_metric(self, metric_name: str) -> dict[str, float]:
        """Calculate specific metric(s). Use `"all"` to get everything."""
        if not self.enabled:
            return {}

        results = {}
        for plugin in self.registry.get_plugins_in_dependency_order():
            if self._should_calculate_plugin(plugin, metric_name):
                window_stats = self._get_plugin_window_stats(plugin)
                results.update(plugin.calculate_realtime(window_stats, self))

        return results

    def calculate_summary_metrics(self) -> dict[str, float]:
        if not self.enabled:
            return {}

        summary = {}

        for plugin in self.active_plugins:
            # For multi-metric plugins, we need to get the right summary data
            if hasattr(plugin, "get_metric_names") and len(plugin.get_metric_names()) > 1:
                # For efficiency plugin, pass empty dict as it accesses summary_metrics directly
                summary_data = {}
            else:
                # For single-metric plugins, pass their specific summary data
                metric_name = plugin.name
                summary_data = self.summary_metrics.get(metric_name, {})

            summary.update(plugin.calculate_summary(summary_data, self))

        return summary

    def reset_window(self):
        if not self.enabled:
            return
        for metric_name in self.metric_windows:
            self.metric_windows[metric_name].clear()
            self.metric_start_times[metric_name] = None
            self.current_batch_data[metric_name] = {"tokens": 0, "samples": 0}

    def reset_all_metrics(self):
        if not self.enabled:
            return
        self.reset_window()
        for metric_name in self.summary_metrics:
            self.summary_metrics[metric_name] = {
                "step_times": [],
                "tokens_per_step": [],
                "samples_per_step": [],
                "step_numbers": [],
            }
