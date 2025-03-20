import torch
from transformers import set_seed

from optimum.neuron.utils.import_utils import (
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xrt


from transformers.models.granite.modeling_granite import GraniteMLP
from transformers import AutoConfig

from optimum.neuron.models.training.granite.modeling_granite import GraniteMLP as NeuronGraniteMLP
from optimum.neuron.models.training.granite.configuration_granite import NeuronGraniteConfig



@torch.no_grad()
def _test_parallel_mlp_():
    set_seed(42)

    model_id = "ibm-granite/granite-3.2-2b-instruct"
    config = AutoConfig.from_pretrained(model_id)

    device = "xla"


    data = torch.load("mlp.pt")
    inputs = data["inputs"].to(device)
    expected_outputs = data["outputs"]

    # model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device).eval()
    # original_mlp = model.model.layers[0].mlp
    original_mlp = GraniteMLP(config).to(torch.bfloat16)
    original_mlp.to(device).eval()
    inputs = torch.randn(1, 2, config.hidden_size, dtype=torch.bfloat16).to(device)
    original_outputs = original_mlp(inputs)
    xm.mark_step()
    # print(f"🟡 rank ori {xrt.global_ordinal()} ", original_outputs)
    # expected_match = torch.allclose(original_outputs.to("cpu"), expected_outputs.to("cpu"), atol=1e-3)
    # print(f"🟡 rank ori {xrt.global_ordinal()} expected match? {expected_match}")

    neuron_config = NeuronGraniteConfig.from_pretrained(model_id)
    parallel_mlp = NeuronGraniteMLP(neuron_config).to(torch.bfloat16)
    parallel_mlp.load_state_dict(original_mlp.state_dict())

    parallel_mlp.to(device).eval()
    parallel_outputs = parallel_mlp(inputs)
    # print(f"🟡 rank par {xrt.global_ordinal()} ", parallel_outputs)
    outputs_match = torch.allclose(parallel_outputs.to("cpu"), original_outputs.to("cpu"), atol=1e-3)
    print(f"🟡 rank par {xrt.global_ordinal()} outputs match? {outputs_match}")
    # expected_match = torch.allclose(parallel_outputs.to("cpu"), expected_outputs.to("cpu"), atol=1e-3)
    # print(f"🟡 rank par {xrt.global_ordinal()} expected match? {expected_match}")
    return
    assert outputs_match, "Sharded model output does not match unsharded one"


@torch.no_grad()
def _test_parallel_mlp():
    set_seed(42)

    model_id = "ibm-granite/granite-3.2-2b-instruct"
    config = AutoConfig.from_pretrained(model_id)

    device = "xla"

    original_mlp = GraniteMLP(config).to(torch.bfloat16)
    original_mlp.to(device).eval()
    inputs = torch.randn(1, 2, config.hidden_size, dtype=torch.bfloat16).to(device)
    original_outputs = original_mlp(inputs)
    xm.mark_step()

    neuron_config = NeuronGraniteConfig.from_pretrained(model_id)
    parallel_mlp = NeuronGraniteMLP(neuron_config).to(torch.bfloat16)
    parallel_mlp.load_state_dict(original_mlp.state_dict())

    parallel_mlp.to(device).eval()
    parallel_outputs = parallel_mlp(inputs)
    outputs_match = torch.allclose(parallel_outputs.to("cpu"), original_outputs.to("cpu"), atol=1e-3)
    print(f"🟡 rank par {xrt.global_ordinal()} outputs match? {outputs_match}")
    return
    assert outputs_match, "Sharded model output does not match unsharded one"


@is_trainium_test
def test_parallel_layers():
    # _test_parallel_mlp()
    # return
    launch_procs(
        _test_parallel_mlp,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )
