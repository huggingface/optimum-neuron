import os

import numpy as np
import pytest
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers.generation.configuration_utils import GenerationConfig

from optimum.neuron.trainers import patch_generation_mixin_to_neuron_generation_mixin
from optimum.neuron.utils.testing_utils import is_trainium_test


def _test_greedy_decoding(model_name, device="cpu", use_cache=False, decoder_only=False):
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
    generation_config.num_beams = 1
    generation_config.do_sample = False
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


testdata = [
    ("t5-small", True, False),
    ("t5-small", False, False),
]


@is_trainium_test
@pytest.mark.parametrize("model_name, use_cache, decoder_only", testdata)
def test_greedy_decoding(model_name, use_cache, decoder_only):
    os.environ["XLA_USE_BF16"] = "0"
    xla_neuron_samples_fp32 = _test_greedy_decoding(model_name=model_name, device="xla", decoder_only=decoder_only)
    os.environ["XLA_USE_BF16"] = "1"
    xla_neuron_samples_bf16 = _test_greedy_decoding(model_name=model_name, device="xla", decoder_only=decoder_only)

    cpu_samples = _test_greedy_decoding(model_name=model_name, device="cpu", decoder_only=decoder_only)

    assert np.array_equal(cpu_samples, xla_neuron_samples_fp32), "XLA Neuron FP32 output doesn't match CPU only output"
    assert np.array_equal(cpu_samples, xla_neuron_samples_bf16), "XLA Neuron bf16 output doesn't match CPU only output"
