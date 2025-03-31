from tempfile import TemporaryDirectory

import pytest

from optimum.neuron.backends.hlo.config import GQA, HloNeuronConfig, Layout
from optimum.neuron.configuration_utils import NeuronConfig


@pytest.mark.parametrize(
    "neuron_config_cls, neuron_config_kwargs",
    [
        (HloNeuronConfig, {}),
        (
            HloNeuronConfig,
            {
                "n_positions": 512,
                "batch_size": 8,
                "amp": "fp16",
                "tp_degree": 4,
                "continuous_batching": True,
                "attention_layout": Layout.BSH,
                "group_query_attention": GQA.REPLICATED_HEADS,
            },
        ),
    ],
    ids=["hlo-default", "hlo-custom"],
)
def test_serialize_neuron_config(neuron_config_cls, neuron_config_kwargs):
    neuron_config = neuron_config_cls(**neuron_config_kwargs)
    for param, value in neuron_config_kwargs.items():
        assert getattr(neuron_config, param) == value
    with TemporaryDirectory() as tmpdir:
        neuron_config.save_pretrained(tmpdir)
        # Verify that we can reload the configuration using the same class
        reloaded_neuron_config = neuron_config_cls.from_pretrained(tmpdir)
        assert reloaded_neuron_config == neuron_config
        # Verify that we can also reload it without knowing the actual class
        auto_neuron_config = NeuronConfig.from_pretrained(tmpdir)
        assert isinstance(auto_neuron_config, neuron_config_cls)
        assert auto_neuron_config == neuron_config
