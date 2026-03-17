import pytest

from optimum.neuron.models.inference.backend.neuron_device_memory import get_neuron_device_memory
from optimum.neuron.utils.system import get_neuron_devices_count


@pytest.mark.skipif(get_neuron_devices_count() < 1, reason="requires a Neuron device")
def test_neuron_device_memory():
    neuron_device_memory = get_neuron_device_memory()
    assert neuron_device_memory is not None

    parsed_device_count = len(neuron_device_memory.devices)
    expected_device_count = get_neuron_devices_count()
    assert parsed_device_count > 0
    # sysfs results are now filtered by /dev-visible device indices,
    # so the counts should match exactly.
    assert parsed_device_count == expected_device_count

    # Ensure each retained device has at least one parsed core.
    assert all(len(device_cores) > 0 for device_cores in neuron_device_memory.devices.values())

    assert neuron_device_memory.get_total_memory() >= 0
