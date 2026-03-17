"""Diagnostic test to verify Neuron device detection on Kubernetes runners."""

import os

from optimum.neuron.utils.neuron_device_memory import get_neuron_device_memory
from optimum.neuron.utils.system import (
    NEURON_DEV_PATTERN,
    get_neuron_device_paths,
    get_neuron_devices_count,
    get_neuron_major,
)


def _get_neuron_minor() -> int:
    """Return the minor device number of /dev/neuron0 (identifies the physical device)."""
    try:
        stat = os.stat("/dev/neuron0")
        return os.minor(stat.st_rdev)
    except FileNotFoundError:
        return -1


def test_neuron_device_detection():
    """Print device detection diagnostics and verify consistency."""
    neuron_major = get_neuron_major()
    neuron_minor = _get_neuron_minor()
    print(f"\nNeuron major number: {neuron_major}")
    print(f"Runner /dev/neuron0 minor: {neuron_minor} (physical device index)")
    print(f"NEURON_RT_VISIBLE_CORES: {os.environ.get('NEURON_RT_VISIBLE_CORES', '<not set>')}")

    # List ALL /dev/neuronX entries with major:minor
    root, _, files = next(os.walk("/dev"))
    all_neuron_files = sorted(f for f in files if NEURON_DEV_PATTERN.match(f))
    print(f"All /dev/neuronX files: {all_neuron_files}")

    for f in all_neuron_files:
        path = f"{root}/{f}"
        try:
            stat = os.stat(path)
            major = os.major(stat.st_rdev)
            minor = os.minor(stat.st_rdev)
            match = "<- MATCH" if major == neuron_major else ""
            print(f"  {path}: major:minor={major}:{minor} {match}")
        except Exception as e:
            print(f"  {path}: ERROR {e}")

    # Verify detection functions from system.py
    device_count = get_neuron_devices_count()
    device_paths = get_neuron_device_paths()
    print(f"get_neuron_devices_count(): {device_count}")
    print(f"get_neuron_device_paths(): {device_paths}")

    # Show device names from neuron_device_memory (sysfs-based detection)
    try:
        device_memory = get_neuron_device_memory()
        print(f"get_neuron_device_memory() devices: {list(device_memory.devices.keys())}")
        for dev_name, cores in device_memory.devices.items():
            print(f"  {dev_name}: cores={list(cores.keys())}")
    except Exception as e:
        print(f"get_neuron_device_memory() ERROR: {e}")

    assert device_count > 0, "Expected at least one Neuron device"
    assert len(device_paths) == device_count, f"Path count ({len(device_paths)}) != device count ({device_count})"
