import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.neuron.models.training.granite.modeling_granite import GraniteForCausalLM
from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
if is_neuronx_distributed_available():
    pass


@torch.no_grad()
def _get_expected_output(model_id, inputs):
    # Get the expected output. Inference will run on CPU, dtype if bfloat16.
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="xla")
    model = model.eval()
    outputs = model(**inputs)
    return outputs.logits.detach()


@torch.no_grad()
def _test_parallel_granite():
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    prompt = "What is Deep Learning?"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to("xla")

    # Expected output is the one loaded from transformers "vanilla" modeling on XLA
    expected_output = _get_expected_output(model_id, inputs)
    print("🔴 No Shard", expected_output, expected_output.shape)
    xm.mark_step()

    # Note that model is init on CPU, then moved  to XLA
    model = GraniteForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device="xla")
    model.eval()
    outputs = model(**inputs)
    xm.mark_step()
    local_rank = xm.get_local_ordinal()
    print(f"🟡 Rank {local_rank}", outputs.logits, outputs.logits.shape)
    outputs_match = torch.allclose(outputs.logits.to("cpu"), expected_output.to("cpu"), atol=1e-3)
    assert outputs_match, "Sharded model output does not match unsharded one"


@is_trainium_test
def test_parallel_granite():
    launch_procs(
        _test_parallel_granite,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )
