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
    """Base class for metrics plugins. Each plugin calculates one type of metric."""
    name: str
    requires_accumulation: bool = False
    depends_on: list[str] | None = None

    @abstractmethod
    def is_enabled(self, args: 'NeuronTrainingArguments') -> bool:
        """Check if this plugin should be active."""
        pass

    @abstractmethod
    def calculate_realtime(self, window_stats: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """Calculate train/ metrics from current window data."""
        pass

    @abstractmethod
    def calculate_summary(self, summary_data: dict, collector: 'TrainingMetricsCollector') -> dict[str, float]:
        """Calculate summary/ metrics from all collected data."""
        pass

    def get_metric_names(self) -> list[str]:
        """Get the metrics this plugin provides. Override for multi-metric plugins."""
        return [self.name]

    def handles_metric(self, metric_name: str) -> bool:
        """Check if this plugin handles the given metric."""
        return metric_name in self.get_metric_names()
