from optimum.neuron import NeuronTracedModel
from optimum.neuron.pipelines import pipeline


def test_export_no_parameters(std_text_task, inf_encoder_model):
    p = pipeline(std_text_task, inf_encoder_model, export=True)
    assert p.task == std_text_task
    assert isinstance(p.model, NeuronTracedModel)
