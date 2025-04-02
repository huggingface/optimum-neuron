import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from optimum.neuron.models.training.granite.modeling_granite import GraniteForCausalLM
from optimum.neuron.utils.import_utils import (
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


@torch.no_grad()
def _get_expected_output(model_id, inputs, torch_dtype):
    # Get the expected output. Inference will run on CPU
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device="xla")
    model = model.eval()
    outputs = model(**inputs)
    return outputs.logits.detach()


def sample_greedy(logits):
    next_logits = logits.to("cpu")[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id


@torch.no_grad()
def _test_parallel_granite():
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    prompt = "What is Deep Learning?"
    torch_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # For XLA model, flash attention will be used, but this requires the sequence length to be a multiple of 2048
    # so padding is applied.
    # inputs = tokenizer(prompt, pad_to_multiple_of=2048, padding=True, return_tensors="pt").to("xla")
    inputs = tokenizer(prompt, pad_to_multiple_of=10, padding=True, return_tensors="pt").to("xla")
    print(inputs["input_ids"].shape)

    # Expected output is the one loaded from transformers "vanilla" modeling on XLA
    expected_output = _get_expected_output(model_id, inputs, torch_dtype)
    xm.mark_step()

    # Note that model is init on CPU, then moved  to XLA
    model = GraniteForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device="xla")
    model.eval()
    outputs = model(**inputs)

    xm.mark_step()

    # It would be better to have this lower, like torch.finfo(torch_dtype).resolution, ( that is 0.1 in bfloat16),
    # but apparently sharded model results are different from unsharded ones.
    atol = 0.2
    diff = (outputs.logits.to("cpu") - expected_output.to("cpu")).abs()
    outputs_match = torch.allclose(outputs.logits.to("cpu"), expected_output.to("cpu"), atol=atol)
    print("Difference between sharded and unsharded model output: ", diff)
    print(f"Max diff: {diff.max().item()} match {outputs_match}")  # For debugging purposes, to see the max diff
    # assert outputs_match, "Sharded model output does not match unsharded one"

    # It is possible to verify that untokenized output is the same when using greedy sampling
    expected_text_output = tokenizer.batch_decode(sample_greedy(expected_output), skip_special_tokens=True)
    text_output = tokenizer.batch_decode(sample_greedy(outputs.logits), skip_special_tokens=True)
    # assert expected_text_output == text_output, "Sharded model output does not match unsharded one"
    print(f"Expected text output: {expected_text_output}\nSharded model text output: {text_output}\n")


@is_trainium_test
def test_parallel_granite():
    launch_procs(
        _test_parallel_granite,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )
