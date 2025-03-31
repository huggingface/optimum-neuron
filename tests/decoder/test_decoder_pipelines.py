import itertools

import pytest
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.pipelines import pipeline
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _test_generation(p):
    assert p.task == "text-generation"
    assert isinstance(p.model, NeuronModelForCausalLM)
    model_batch_size = getattr(p.model.config, "neuron")["batch_size"]
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
    p = pipeline("text-generation", "Qwen/Qwen2.5-0.5B", export=True)
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_load_no_parameters(neuron_decoder_path):
    p = pipeline("text-generation", neuron_decoder_path)
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_from_model_and_tokenizer(neuron_decoder_path):
    m = NeuronModelForCausalLM.from_pretrained(neuron_decoder_path)
    t = AutoTokenizer.from_pretrained(neuron_decoder_path)
    p = pipeline("text-generation", model=m, tokenizer=t)
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_error_already_exported(neuron_decoder_path):
    with pytest.raises(ValueError, match="already been exported"):
        pipeline("text-generation", neuron_decoder_path, export=True)


@is_inferentia_test
@requires_neuronx
def test_error_needs_export():
    with pytest.raises(ValueError, match="must be exported"):
        pipeline("text-generation", "Qwen/Qwen2.5-0.5B", export=False)


@is_inferentia_test
@requires_neuronx
def test_from_hub(neuron_decoder_config):
    model_id = neuron_decoder_config["neuron_model_id"]
    p = pipeline("text-generation", model_id)
    _test_generation(p)
