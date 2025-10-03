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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .collector import TrainingMetricsCollector
    from ..training_args import NeuronTrainingArguments


@dataclass
class MetricPlugin(ABC):
    """
    Base class for training metrics plugins.

    Each plugin is responsible for calculating a specific type of metric
    (e.g., throughput, MFU, efficiency) from collected timing data.
    """
    name: str
    requires_accumulation: bool = False
    depends_on: list[str] | None = None

    @abstractmethod
    def is_enabled(self, args: 'NeuronTrainingArguments') -> bool:
        """
        Check if plugin should be active based on training arguments.

        Args:
            args: Training arguments containing metric enable flags

        Returns:
            True if the plugin should be enabled
        """
        pass

    @abstractmethod
    def calculate_realtime(self, window_stats: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """
        Calculate real-time train/ metrics from moving window data.

        Args:
            window_stats: Statistics from the metric's moving window
            collector: Reference to the metrics collector for inter-plugin communication

        Returns:
            Dictionary of train/ prefixed metrics
        """
        pass

    @abstractmethod
    def calculate_summary(self, summary_data: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """
        Calculate end-of-training summary/ metrics from all collected data.

        Args:
            summary_data: All collected timing data for this metric
            collector: Reference to the metrics collector for inter-plugin communication

        Returns:
            Dictionary of summary/ prefixed metrics
        """
        pass

    def get_metric_names(self) -> list[str]:
        """
        Get list of metric names this plugin handles.

        Override this for plugins that handle multiple metrics.
        Default implementation returns the plugin name.

        Returns:
            List of metric names handled by this plugin
        """
        return [self.name]

    def handles_metric(self, metric_name: str) -> bool:
        """
        Check if this plugin handles the given metric name.

        Args:
            metric_name: Name of the metric to check

        Returns:
            True if this plugin handles the metric
        """
        return metric_name in self.get_metric_names()