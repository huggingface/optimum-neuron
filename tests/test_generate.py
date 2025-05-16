import os
from unittest import TestCase

import numpy as np
import pytest
from parameterized import parameterized
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.generation.configuration_utils import GenerationConfig

from optimum.neuron.trainers import patch_generation_mixin_to_neuron_generation_mixin
from optimum.neuron.utils.testing_utils import is_trainium_test

from .utils import TrainiumTestMixin


def _test_generative_decoding(
    model_name,
    device="cpu",
    use_cache=False,
    decoder_only=False,
    generation_config_update={"num_beams": 1, "do_sample": False},
):
    if device == "xla":
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = (
        AutoModelForCausalLM.from_pretrained(model_name).to(device)
        if decoder_only
        else AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    )
    model_config = AutoConfig.from_pretrained(model_name)
    generation_config = GenerationConfig.from_model_config(model_config)

    if decoder_only:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "left"
        generation_config.pad_token_id = generation_config.eos_token_id

    # Set to greedy search
    generation_config.update(**generation_config_update)
    generation_config.use_cache = use_cache

    patch_generation_mixin_to_neuron_generation_mixin(model)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    task_prompt = "translate English to German: "
    input_strings = [
        "How old are you?",
        "What is your name?",
        "We work at Amazon",
        "I am trying to test the generation method",
    ]

    results = []
    if device == "xla":
        xm.mark_step()
    for input_str in input_strings:
        input_ids = tokenizer(task_prompt + input_str, return_tensors="pt", padding="max_length", max_length=50).to(
            device
        )
        outputs = model.generate(**input_ids, max_new_tokens=20, use_cache=False, generation_config=generation_config)
        outputs = outputs.detach().cpu().numpy()
        results.append(outputs)

    return np.array(results)


GREEDY_TESTDATA = [
    ("t5-small", True, False, ""),
    ("t5-small", False, False, ""),
]

BEAM_SEARCH_TESTDATA = [
    ("facebook/bart-base", False, False, "--model-type=transformer --enable-saturate-infinity"),
    ("t5-small", False, False, "--model-type=transformer"),
    ("t5-small", True, False, "--model-type=transformer"),
]


@is_trainium_test
class GenerateTestCase(TrainiumTestMixin, TestCase):
    @parameterized.expand(GREEDY_TESTDATA)
    @pytest.mark.skip("Remove once generate fix (#262) has been merged.")
    def test_greedy_decoding(self, model_name, use_cache, decoder_only, compiler_flags):
        os.environ["NEURON_CC_FLAGS"] = compiler_flags
        os.environ["XLA_USE_BF16"] = "0"
        xla_neuron_samples_fp32 = _test_generative_decoding(
            model_name=model_name, device="xla", decoder_only=decoder_only
        )
        os.environ["XLA_USE_BF16"] = "1"
        xla_neuron_samples_bf16 = _test_generative_decoding(
            model_name=model_name, device="xla", decoder_only=decoder_only
        )

        cpu_samples = _test_generative_decoding(model_name=model_name, device="cpu", decoder_only=decoder_only)

        assert np.array_equal(
            cpu_samples, xla_neuron_samples_fp32
        ), "XLA Neuron FP32 output doesn't match CPU only output"
        assert np.array_equal(
            cpu_samples, xla_neuron_samples_bf16
        ), "XLA Neuron bf16 output doesn't match CPU only output"

    @parameterized.expand(BEAM_SEARCH_TESTDATA)
    @pytest.mark.skip("Remove once generate fix (#262) has been merged.")
    def test_beam_search_decoding(self, model_name, use_cache, decoder_only, compiler_flags):
        os.environ["NEURON_CC_FLAGS"] = compiler_flags
        config_update = {"num_beams": 4, "min_length": 21, "max_length": 21}

        os.environ["XLA_USE_BF16"] = "0"
        xla_neuron_samples_fp32 = _test_generative_decoding(
            model_name=model_name, device="xla", decoder_only=decoder_only, generation_config_update=config_update
        )
        os.environ["XLA_USE_BF16"] = "1"
        xla_neuron_samples_bf16 = _test_generative_decoding(
            model_name=model_name, device="xla", decoder_only=decoder_only, generation_config_update=config_update
        )

        cpu_samples = _test_generative_decoding(
            model_name=model_name, device="cpu", decoder_only=decoder_only, generation_config_update=config_update
        )

        assert np.array_equal(
            cpu_samples, xla_neuron_samples_fp32
        ), "XLA Neuron FP32 output doesn't match CPU only output"
        assert np.array_equal(
            cpu_samples, xla_neuron_samples_bf16
        ), "XLA Neuron bf16 output doesn't match CPU only output"
