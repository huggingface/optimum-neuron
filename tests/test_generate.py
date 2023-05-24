import os

import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from optimum.neuron.trainers import patch_generation_mixin_to_neuron_generation_mixin


def _create_samples(model_name="t5-small", device="cpu", use_cache=False):
    if device == "xla":
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    patch_generation_mixin_to_neuron_generation_mixin(model)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    task_prompt = "translate English to German: "
    input_strings = [
        "How old are you?",
        "What is your name?",
        "We work at Amazon",
        "I am trying to test the generation method"
    ]

    results = []
    if device == "xla":
        xm.mark_step()
    for input_str in input_strings:
        encoder_input_ids = tokenizer(task_prompt + input_str, return_tensors="pt", padding="max_length",
                                      max_length=50).to(device)
        outputs = model.generate(
            **encoder_input_ids, max_new_tokens=20, use_cache=False
        )
        outputs = outputs.detach().cpu().numpy()
        if device == "cpu":
            outputs = np.pad(outputs, ((0, 0), (0, 21 - outputs.shape[-1])))
        results.append(outputs)

    return np.array(results)


def test_greedy():
    os.environ['XLA_USE_BF16'] = '0'
    xla_neuron_samples_fp32 = _create_samples(device="xla")
    os.environ['XLA_USE_BF16'] = '1'
    xla_neuron_samples_bf16 = _create_samples(device="xla")

    cpu_samples = _create_samples(device="cpu")

    assert np.array_equal(cpu_samples, xla_neuron_samples_fp32), "XLA Neuron FP32 output doesn't match CPU only output"
    assert np.array_equal(cpu_samples, xla_neuron_samples_bf16), "XLA Neuron bf16 output doesn't match CPU only output"
