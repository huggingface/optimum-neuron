import subprocess
from tempfile import TemporaryDirectory

import pytest

from optimum.neuron.backends.hlo.config import HloNeuronConfig
from optimum.neuron.configuration_utils import NeuronConfig
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("batch_size, sequence_length, auto_cast_type", [[1, 512, "fp16"], [2, 128, "bf16"]])
@pytest.mark.parametrize("num_cores", [1, 2])
def test_export_decoder_cli(batch_size, sequence_length, auto_cast_type, num_cores):
    model_id = "llamafactory/tiny-random-Llama-3"
    with TemporaryDirectory() as tempdir:
        subprocess.run(
            [
                "optimum-cli",
                "export",
                "neuron",
                "--model",
                model_id,
                "--sequence_length",
                f"{sequence_length}",
                "--batch_size",
                f"{batch_size}",
                "--auto_cast_type",
                auto_cast_type,
                "--num_cores",
                f"{num_cores}",
                "--task",
                "text-generation",
                tempdir,
            ],
            shell=False,
            check=True,
        )
        # Check exported config
        neuron_config = NeuronConfig.from_pretrained(tempdir)
        assert isinstance(neuron_config, HloNeuronConfig)
        assert neuron_config.batch_size == batch_size
        assert neuron_config.sequence_length == sequence_length
        assert neuron_config.auto_cast_type == auto_cast_type
        assert neuron_config.tp_degree == num_cores
        assert neuron_config.checkpoint_id == model_id
