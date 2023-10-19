import pytest

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.pipelines import pipeline
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _test_generation(p):
    assert p.task == "text-generation"
    assert isinstance(p.model, NeuronModelForCausalLM)
    batch_size = getattr(p.model.config, "neuron")["batch_size"]
    prompt = "I like you."
    prompts = [prompt] * batch_size
    outputs = p(prompts, do_sample=True, top_k=50, top_p=0.9, temperature=0.9)
    assert len(outputs) == batch_size
    for output in outputs:
        # We only ever generate one sequence per sample
        sequence = output[0]
        assert sequence["generated_text"].startswith(prompt)


@is_inferentia_test
@requires_neuronx
def test_export_no_parameters(inf_decoder_model):
    p = pipeline("text-generation", inf_decoder_model, export=True)
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_load_no_parameters(inf_decoder_path):
    p = pipeline("text-generation", inf_decoder_path)
    _test_generation(p)


@is_inferentia_test
@requires_neuronx
def test_error_already_exported(inf_decoder_path):
    with pytest.raises(ValueError, match="already been exported"):
        pipeline("text-generation", inf_decoder_path, export=True)


@is_inferentia_test
@requires_neuronx
def test_error_needs_export(inf_decoder_model):
    with pytest.raises(ValueError, match="must be exported"):
        pipeline("text-generation", inf_decoder_model, export=False)
