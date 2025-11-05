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

from .base import MetricPlugin


class PluginRegistry:
    """Manages plugin discovery and provides explicit inter-plugin communication."""

    def __init__(self, plugins: list[MetricPlugin]):
        self.plugins = {p.name: p for p in plugins}
        self.metric_to_plugin = {}

        # Build reverse lookup: metric_name -> plugin
        for plugin in plugins:
            for metric_name in plugin.get_metric_names():
                self.metric_to_plugin[metric_name] = plugin

    def get_plugin(self, plugin_name: str) -> MetricPlugin | None:
        """Get plugin by name."""
        return self.plugins.get(plugin_name)

    def get_plugin_for_metric(self, metric_name: str) -> MetricPlugin | None:
        """Get the plugin that handles a specific metric."""
        return self.metric_to_plugin.get(metric_name)

    def validate_dependencies(self) -> None:
        """Make sure all plugin dependencies are satisfied."""
        for plugin in self.plugins.values():
            if not plugin.depends_on:
                continue

            for dep_metric in plugin.depends_on:
                if dep_metric not in self.metric_to_plugin:
                    raise ValueError(f"Plugin '{plugin.name}' needs metric '{dep_metric}', but no plugin provides it")

    def get_plugins_in_dependency_order(self) -> list[MetricPlugin]:
        """Sort plugins so dependencies come first."""
        independent = [p for p in self.plugins.values() if not p.depends_on]
        dependent = [p for p in self.plugins.values() if p.depends_on]
        return independent + dependent
