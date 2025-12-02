import pytest

from optimum.neuron.utils.neuron_device_memory import get_neuron_device_memory
from optimum.neuron.utils.system import get_neuron_devices_count


@pytest.mark.skipif(get_neuron_devices_count() < 1, reason="requires a Neuron device")
def test_neuron_device_memory():
    neuron_device_memory = get_neuron_device_memory()
    assert neuron_device_memory is not None
    assert len(neuron_device_memory.devices) == get_neuron_devices_count()
    assert neuron_device_memory.get_total_memory() >= 0
