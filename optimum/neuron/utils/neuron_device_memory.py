#!/usr/bin/env python3
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
"""
Script to read Neuron device memory information using sysfs.

This script reads device memory statistics for all Neuron cores from the sysfs
filesystem at /sys/devices/virtual/neuron_device/ and returns a dictionary
containing device memory breakdown for each core.

Based on: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-sysfs-user-guide.html
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


SYSFS_NEURON_BASE = Path("/sys/devices/virtual/neuron_device")

# Device memory categories as per the documentation
DEVICE_MEM_CATEGORIES = [
    "collectives",
    "constants",
    "dma_rings",
    "driver_memory",
    "model_code",
    "model_shared_scratchpad",
    "nonshared_scratchpad",
    "notifications",
    "runtime_memory",
    "tensors",
    "uncategorized",
]


def read_sysfs_value(file_path: Path) -> str:
    """
    Read a single value from a sysfs file.

    Args:
        file_path: Path to the sysfs file

    Returns:
        The content of the file as a string (stripped of whitespace)

    """
    with open(file_path, "r") as f:
        return f.read().strip()


@dataclass
class MemoryCounter:
    total: Optional[int]
    present: Optional[int]
    peak: Optional[int]


@dataclass
class CoreDeviceMemory:
    categories: Dict[str, MemoryCounter]

    def total_memory(self) -> int:
        total = 0
        for c in self.categories.values():
            if c.total is not None:
                total += c.total
        return total


def read_memory_counter(base_path: Path, category: str) -> MemoryCounter:
    """
    Read total, present, and peak values for a memory counter.

    Args:
        base_path: Base path to the memory category directory
        category: Memory category name (e.g., 'constants', 'tensors')

    Returns:
        MemoryCounter with 'total', 'present', and 'peak' values
    """
    counter_path = base_path / category

    def read_values(labels: list[str]) -> list[int]:
        values = []
        for label in labels:
            try:
                value = int(read_sysfs_value(counter_path / label))
            except (FileNotFoundError, PermissionError, ValueError):
                value = None
            values.append(value)
        return values

    return MemoryCounter(*read_values(["total", "present", "peak"]))


def read_core_device_memory(core_path: Path) -> CoreDeviceMemory:
    """
    Read device memory statistics for a single NeuronCore.

    Args:
        core_path: Path to the neuron_core directory

    Returns:
        CoreDeviceMemory containing per-category counters
    """
    device_mem_path = core_path / "stats" / "memory_usage" / "device_mem"

    if not device_mem_path.exists():
        return CoreDeviceMemory(categories={})

    categories: Dict[str, MemoryCounter] = {}

    for category in DEVICE_MEM_CATEGORIES:
        try:
            categories[category] = read_memory_counter(device_mem_path, category)
        except Exception:
            # Skip categories that can't be read
            continue

    return CoreDeviceMemory(categories=categories)


class NeuronDeviceMemory:
    """
    Class representing Neuron device memory usage across all devices and cores.

    Attributes:
        devices: Dictionary mapping device names to their core memory statistics
    """

    def __init__(self, devices: Dict[str, Dict[str, CoreDeviceMemory]]):
        """
        Initialize NeuronDeviceMemory.

        Args:
            devices: Dictionary mapping device names to their core memory statistics
        """
        self.devices = devices

    def get_total_memory(self) -> int:
        """
        Calculate the total device memory used across all Neuron devices and cores.

        Returns:
            Total device memory in bytes
        """
        total_memory = 0
        for cores in self.devices.values():
            for core_mem in cores.values():
                total_memory += core_mem.total_memory()
        return total_memory

    def _format_memory(self, memory_bytes: int) -> str:
        """
        Format memory bytes into a human-readable string.

        Args:
            memory_bytes: Memory size in bytes

        Returns:
            Formatted string (e.g., "1.23 GB", "456.78 MB")
        """
        for factor, suffix in [
            (2**40, "TB"),
            (2**30, "GB"),
            (2**20, "MB"),
            (2**10, "KB"),
        ]:
            if memory_bytes >= factor:
                return f"{memory_bytes / factor:.2f} {suffix}"
        return f"{memory_bytes} bytes"

    def __str__(self) -> str:
        """
        Return a string with only the total device memory formatted.

        Returns:
            String like "Neuron device memory usage: 1.23 GB"
        """
        total_memory = self.get_total_memory()
        memory_str = self._format_memory(total_memory)
        return f"Neuron device memory usage: {memory_str}"

    def __repr__(self) -> str:
        """
        Return a detailed string with information per device and core.

        Returns:
            Multi-line string with detailed memory breakdown per device and core
        """
        lines = ["Neuron Device Memory Information", "=" * 80]

        for device_name, cores in self.devices.items():
            lines.append(f"\n{device_name}:")
            for core_name, core_mem in cores.items():
                lines.append(f"  {core_name}:")

                # Calculate total device memory used for this core
                total_mem = 0
                for category, counters in core_mem.categories.items():
                    if counters.total is not None:
                        total_mem += counters.total
                        lines.append(
                            f"    {category:30s}: {counters.total:>12,d} bytes (peak: {(counters.peak or 0):>12,d})"
                        )

                lines.append(f"    {'TOTAL':30s}: {total_mem:>12,d} bytes")

        return "\n".join(lines)


def get_neuron_device_memory() -> NeuronDeviceMemory:
    """
    Read device memory information for all Neuron devices and cores.

    Returns:
        NeuronDeviceMemory object containing memory statistics for all devices and cores
    """
    if not SYSFS_NEURON_BASE.exists():
        raise RuntimeError(
            f"Neuron sysfs not found at {SYSFS_NEURON_BASE}. Make sure the Neuron driver is installed and loaded."
        )

    devices: Dict[str, Dict[str, CoreDeviceMemory]] = {}

    # Iterate through all neuron devices (neuron0, neuron1, etc.)
    for device_dir in sorted(SYSFS_NEURON_BASE.glob("neuron*")):
        if not device_dir.is_dir():
            continue

        device_name = device_dir.name
        devices[device_name] = {}

        # Iterate through all cores in this device (neuron_core0, neuron_core1, etc.)
        for core_dir in sorted(device_dir.glob("neuron_core*")):
            if not core_dir.is_dir():
                continue

            core_name = core_dir.name
            core_mem = read_core_device_memory(core_dir)
            if core_mem.categories:  # Only add if we got some data
                devices[device_name][core_name] = core_mem

    return NeuronDeviceMemory(devices)


def main():
    """
    Main function to demonstrate usage.
    """
    try:
        device_memory = get_neuron_device_memory()

        # Print detailed information using repr()
        print(repr(device_memory))

        # Print total memory using str()
        print(str(device_memory))

        return device_memory

    except RuntimeError as e:
        print(f"Error: {e}")
        return None
    except PermissionError:
        print("Error: Permission denied. Try running with sudo.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


if __name__ == "__main__":
    main()
