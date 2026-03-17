import os

import pytest

from optimum.neuron.utils import neuron_device_memory
from optimum.neuron.utils.neuron_device_memory import get_neuron_device_memory
from optimum.neuron.utils.system import get_neuron_devices_count


@pytest.mark.skipif(get_neuron_devices_count() < 1, reason="requires a Neuron device")
def test_neuron_device_memory():
    neuron_device_memory = get_neuron_device_memory()
    assert neuron_device_memory is not None
    assert len(neuron_device_memory.devices) == get_neuron_devices_count()
    assert neuron_device_memory.get_total_memory() >= 0


def test_neuron_device_memory_filters_non_visible_devices(tmp_path, monkeypatch):
    # Validate that get_neuron_device_memory only reports sysfs devices marked as visible.
    sysfs_root = tmp_path / "neuron_device"
    monkeypatch.setattr(neuron_device_memory, "SYSFS_NEURON_BASE", sysfs_root)
    monkeypatch.setattr(
        neuron_device_memory,
        "_get_visible_neuron_device_names",
        lambda: {"neuron0"},
    )

    def write_counter(device_name: str, core_name: str, category: str, total: int, present: int, peak: int):
        category_path = sysfs_root / device_name / core_name / "stats" / "memory_usage" / "device_mem" / category
        category_path.mkdir(parents=True, exist_ok=True)
        (category_path / "total").write_text(str(total), encoding="utf-8")
        (category_path / "present").write_text(str(present), encoding="utf-8")
        (category_path / "peak").write_text(str(peak), encoding="utf-8")

    write_counter("neuron0", "neuron_core0", "constants", 1024, 512, 2048)
    write_counter("neuron1", "neuron_core0", "constants", 2048, 1024, 4096)

    result = get_neuron_device_memory()

    assert set(result.devices.keys()) == {"neuron0"}
    assert result.get_total_memory() == 1024


def test_neuron_device_memory_maps_visible_devices_by_major_minor(tmp_path, monkeypatch):
    # Validate that visible /dev nodes are mapped to sysfs devices by major:minor,
    # not by the /dev filename (which may be remapped in containers).
    sysfs_root = tmp_path / "neuron_device"
    monkeypatch.setattr(neuron_device_memory, "SYSFS_NEURON_BASE", sysfs_root)

    def write_counter(device_name: str, total: int):
        category_path = (
            sysfs_root / device_name / "neuron_core0" / "stats" / "memory_usage" / "device_mem" / "constants"
        )
        category_path.mkdir(parents=True, exist_ok=True)
        (category_path / "total").write_text(str(total), encoding="utf-8")
        (category_path / "present").write_text("0", encoding="utf-8")
        (category_path / "peak").write_text("0", encoding="utf-8")

    # sysfs has two physical devices with distinct major:minor IDs.
    (sysfs_root / "neuron0").mkdir(parents=True, exist_ok=True)
    (sysfs_root / "neuron0" / "dev").write_text("238:0", encoding="utf-8")
    write_counter("neuron0", 1024)

    (sysfs_root / "neuron1").mkdir(parents=True, exist_ok=True)
    (sysfs_root / "neuron1" / "dev").write_text("238:1", encoding="utf-8")
    write_counter("neuron1", 2048)

    # Simulate container remap: /dev only exposes a node named neuron0,
    # but it points to physical device 238:1 (sysfs neuron1).
    fake_dev_root = tmp_path / "dev"
    fake_dev_root.mkdir(parents=True, exist_ok=True)
    fake_dev_node = fake_dev_root / "neuron0"
    fake_dev_node.write_text("", encoding="utf-8")

    original_iterdir = neuron_device_memory.Path.iterdir
    original_is_char_device = neuron_device_memory.Path.is_char_device
    original_stat = neuron_device_memory.Path.stat

    class _FakeDeviceStat:
        st_rdev = os.makedev(238, 1)

    def fake_iterdir(self):
        if self == neuron_device_memory.Path("/dev"):
            return iter([fake_dev_node])
        return original_iterdir(self)

    def fake_is_char_device(self):
        if self == fake_dev_node:
            return True
        return original_is_char_device(self)

    def fake_stat(self):
        if self == fake_dev_node:
            return _FakeDeviceStat()
        return original_stat(self)

    monkeypatch.setattr(neuron_device_memory.Path, "iterdir", fake_iterdir)
    monkeypatch.setattr(neuron_device_memory.Path, "is_char_device", fake_is_char_device)
    monkeypatch.setattr(neuron_device_memory.Path, "stat", fake_stat)

    result = get_neuron_device_memory()

    assert set(result.devices.keys()) == {"neuron1"}
    assert result.get_total_memory() == 2048
