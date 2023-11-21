from optimum.neuron import NeuronBaseModel
from optimum.neuron.pipelines import pipeline
from optimum.neuron.utils.testing_utils import is_inferentia_test


@is_inferentia_test
def test_export_no_parameters(std_text_task, inf_encoder_model):
    p = pipeline(std_text_task, inf_encoder_model, export=True)
    assert p.task == std_text_task
    assert isinstance(p.model, NeuronBaseModel)
