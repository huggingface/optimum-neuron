from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.pipelines import pipeline
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
def test_export_no_parameters(inf_decoder_model):
    p = pipeline("text-generation", inf_decoder_model, export=True)
    assert p.task == "text-generation"
    assert isinstance(p.model, NeuronModelForCausalLM)
