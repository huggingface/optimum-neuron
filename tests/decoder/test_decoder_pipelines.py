import itertools
import os

import pytest
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.pipelines import pipeline
from optimum.neuron.utils import DTYPE_MAPPER
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _test_generation(p):
    assert p.task == "text-generation"
    assert isinstance(p.model, NeuronModelForCausalLM)
    model_batch_size = p.model.neuron_config.batch_size
    prompt = "I like you."
    # We check the ability of the pipeline to split the inputs by using different
    # combinations of input_size and batch_size
    input_sizes = [model_batch_size, model_batch_size * 2]
    batch_sizes = [model_batch_size]
    if model_batch_size > 1:
        batch_sizes.append(model_batch_size // 2)
    for input_size, batch_size, return_tensors in itertools.product(input_sizes, batch_sizes, [True, None]):
        prompts = [prompt] * input_size
        outputs = p(
            prompts,
            return_tensors=return_tensors,
            batch_size=batch_size,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.9,
        )
        assert len(outputs) == input_size
        for input, output in zip(prompts, outputs):
            # We only ever generate one sequence per input
            sequence = output[0]
            if return_tensors:
                input_ids = p.tokenizer(input).input_ids
                assert sequence["generated_token_ids"][: len(input_ids)] == input_ids
            else:
                assert sequence["generated_text"].startswith(input)


@is_inferentia_test
@requires_neuronx
def test_export_no_parameters():
    visible_cores = os.environ.get("NEURON_RT_NUM_CORES", None)
    os.environ["NEURON_RT_NUM_CORES"] = "2"
    p = pipeline("text-generation", "Qwen/Qwen2.5-0.5B", export=True)
    _test_generation(p)
    if visible_cores is None:
        os.environ.pop("NEURON_RT_NUM_CORES", None)
    else:
        os.environ["NEURON_RT_NUM_CORES"] = visible_cores


@is_inferentia_test
@requires_neuronx
def test_export_parameters():
    export_kwargs = {"batch_size": 2, "sequence_length": 1024, "num_cores": 2, "auto_cast_type": "fp16"}
    p = pipeline("text-generation", "Qwen/Qwen2.5-0.5B", export=True, **export_kwargs)
    for key, value in export_kwargs.items():
        if key == "num_cores":
            key = "tp_degree"
        if key == "auto_cast_type":
            key = "torch_dtype"
            if isinstance(value, str):
                value = DTYPE_MAPPER.pt(value)
        assert getattr(p.model.neuron_config, key) == value
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_load_no_parameters(base_neuron_decoder_path):
    p = pipeline("text-generation", base_neuron_decoder_path)
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_from_model_and_tokenizer(base_neuron_decoder_path):
    m = NeuronModelForCausalLM.from_pretrained(base_neuron_decoder_path)
    t = AutoTokenizer.from_pretrained(base_neuron_decoder_path)
    p = pipeline("text-generation", model=m, tokenizer=t)
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_error_already_exported(base_neuron_decoder_path):
    with pytest.raises(ValueError, match="already been exported"):
        pipeline("text-generation", base_neuron_decoder_path, export=True)


@is_inferentia_test
@requires_neuronx
def test_error_needs_export():
    with pytest.raises(ValueError, match="must be exported"):
        pipeline("text-generation", "Qwen/Qwen2.5-0.5B", export=False)


@is_inferentia_test
@requires_neuronx
def test_from_hub(base_neuron_decoder_config):
    model_id = base_neuron_decoder_config["neuron_model_id"]
    p = pipeline("text-generation", model_id)
    _test_generation(p)
