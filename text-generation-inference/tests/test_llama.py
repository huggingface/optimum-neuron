from tempfile import TemporaryDirectory

import pytest
from helpers import check_decode_multiple, check_decode_single, check_prefill
from text_generation_server.generator import NeuronGenerator
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM


MODEL_ID = "princeton-nlp/Sheared-LLaMA-1.3B"
BATCH_SIZE = 4
SEQUENCE_LENGTH = 1024
NUM_CORES = 2


@pytest.fixture(scope="module")
def model_path():
    with TemporaryDirectory() as tmpdir:
        AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(tmpdir)
        model = NeuronModelForCausalLM.from_pretrained(
            MODEL_ID, export=True, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, num_cores=NUM_CORES
        )
        model.save_pretrained(tmpdir)
        yield tmpdir


def test_info(model_path):
    generator = NeuronGenerator.from_pretrained(model_path)
    info = generator.info
    assert info.requires_padding is True
    assert info.device_type == "xla"
    assert info.window_size == 0
    assert info.speculate == 0


@pytest.mark.parametrize(
    "input_text, token_id, token_text, do_sample",
    [
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            13,
            "\n",
            False,
        ],
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            739,
            " It",
            True,
        ],
    ],
    ids=["greedy", "sample"],
)
@pytest.mark.parametrize("batch_size", [1, 4], ids=["single", "multiple"])
def test_prefill(input_text, token_id, token_text, do_sample, batch_size, model_path):
    check_prefill(input_text, token_id, token_text, do_sample, batch_size, model_path)


@pytest.mark.parametrize(
    "input_text, max_new_tokens, generated_text, do_sample",
    [
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            20,
            "\nThe sun was shining, and the birds were singing.\nThe sun was shining,",
            False,
        ],
        [
            "It was a bright cold day in April, and the clocks were striking thirteen.",
            20,
            " It was time to return to the village of Uckfield to face the fury of the evil",
            True,
        ],
    ],
    ids=["greedy", "sample"],
)
def test_decode_single(input_text, max_new_tokens, generated_text, do_sample, model_path):
    check_decode_single(input_text, max_new_tokens, generated_text, do_sample, model_path)


def test_decode_multiple(model_path):
    check_decode_multiple(model_path)
