import subprocess
from tempfile import TemporaryDirectory

import pytest

from optimum.neuron.configuration_utils import NeuronConfig
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.utils import DTYPE_MAPPER
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("batch_size, sequence_length, auto_cast_type", [[1, 512, "fp16"], [2, 128, "bf16"]])
@pytest.mark.parametrize("instance_type", ["trn1", "trn2"])
def test_export_decoder_cli(batch_size: int, sequence_length: int, auto_cast_type: str, instance_type: str):
    model_id = "llamafactory/tiny-random-Llama-3"
    with TemporaryDirectory() as tempdir:
        subprocess.run(
            [
                "optimum-cli",
                "export",
                "neuron",
                "--model",
                model_id,
                "--instance_type",
                instance_type,
                "--sequence_length",
                f"{sequence_length}",
                "--batch_size",
                f"{batch_size}",
                "--auto_cast_type",
                auto_cast_type,
                "--tensor_parallel_size",
                "2",
                "--task",
                "text-generation",
                tempdir,
            ],
            shell=False,
            check=True,
        )
        # Check exported config
        neuron_config = NeuronConfig.from_pretrained(tempdir)
        assert isinstance(neuron_config, NxDNeuronConfig)
        assert neuron_config.batch_size == batch_size
        assert neuron_config.sequence_length == sequence_length
        assert neuron_config.torch_dtype == DTYPE_MAPPER.pt(auto_cast_type)
        assert neuron_config.target == instance_type
        assert neuron_config.checkpoint_id == model_id
