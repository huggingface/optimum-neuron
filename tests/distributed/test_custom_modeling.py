# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import math
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.utils._pytree as pytree
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import optimum
import optimum.neuron.models.training
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.utils.import_utils import (
    is_neuronx_available,
    is_neuronx_distributed_available,
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs
from ..utils import SEED, StaticSeedPatcher, create_accelerator, get_model_inputs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_pipeline_model_parallel_rank,
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
    from neuronx_distributed.utils.model_utils import move_model_to_device


from optimum.neuron.models.training.granite.modeling_granite import GraniteForCausalLM
from optimum.neuron.utils.import_utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_available():
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
    from neuronx_distributed.utils.model_utils import move_model_to_device


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
    inputs = tokenizer(prompt, return_tensors="pt").to("xla")

    # Expected output is the one loaded from transformers "vanilla" modeling on XLA
    expected_output = _get_expected_output(model_id, inputs, torch_dtype)
    xm.mark_step()

    # Note that model is init on CPU, then moved  to XLA
    tp_size = get_tensor_model_parallel_size()
    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        sequence_parallel_enabled=False,
    )
    model = GraniteForCausalLM.from_pretrained(model_id, trn_config, torch_dtype=torch_dtype)
    move_model_to_device(model, xm.xla_device())
    model.eval()
    outputs = model(**inputs)
    xm.mark_step()

    # It would be better to have this lower, like torch.finfo(torch_dtype).resolution, ( that is 0.1 in bfloat16),
    # but apparently sharded model results are different from unsharded ones.
    atol = 0.2
    outputs_match = torch.allclose(outputs.logits.to("cpu"), expected_output.to("cpu"), atol=atol)
    assert outputs_match, "Sharded model output does not match unsharded one"

    # It is possible to verify that untokenized output is the same when using greedy sampling
    expected_text_output = tokenizer.batch_decode(sample_greedy(expected_output), skip_special_tokens=True)
    text_output = tokenizer.batch_decode(sample_greedy(outputs.logits), skip_special_tokens=True)
    assert expected_text_output == text_output, "Sharded model output does not match unsharded one"


@is_trainium_test
def test_parallel_granite():
    launch_procs(
        _test_parallel_granite,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )


def _custom_model_matches_original_model(
    model_class_name,
    model_name_or_path,
    parallel_sizes,
    sequence_parallel_enabled,
    qkv_implementation,
    attn_implementation,
    monkeypatch,
    # It is tricky to test for `torch_dtype=torch.bfloat16` because the precision is low and the "error" induced by
    # the parallel linears accumulates over the layers.
    torch_dtype=torch.float32,
):
    monkeypatch.setattr(
        optimum.neuron.models.training.loss_utils, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True
    )

    world_size, tp_size, pp_size = parallel_sizes
    dp_size = world_size // (tp_size * pp_size)
    pp_rank = get_pipeline_model_parallel_rank()

    static_seed_patcher = StaticSeedPatcher(SEED)

    accelerator = create_accelerator(
        tp_size, pp_size, parallelize_embeddings=False, sequence_parallel_enabled=sequence_parallel_enabled
    )

    orig_model_class = getattr(transformers, model_class_name)
    with static_seed_patcher:
        orig_model = orig_model_class.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)

    # It is ok to use this accelerator because `patch_model_for_neuron` does not depend on the TP or PP size.
    orig_model = accelerator.patch_model_for_neuron(orig_model)

    # Since the new KV cache system it seems that if orig_model.use_cache != model.use_cache, the losses between
    # the two models will not match. It either comes from Transformers itself or Optimum Neuron.
    # TODO: investigate this.
    if pp_size == 1:
        orig_model.config.use_cache = True
    else:
        orig_model.config.use_cache = False
    move_model_to_device(orig_model, xm.xla_device())
    orig_model = orig_model.eval()

    if pp_size > 1:
        pytest.skip(f"Pipeline parallelism is not supported for {model_class_name}.")

    if sequence_parallel_enabled and attn_implementation == "flash_attention_2":
        pad_to_multiple_of = (2048 * tp_size) // math.gcd(2048, tp_size)
    elif sequence_parallel_enabled:
        pad_to_multiple_of = tp_size
    elif attn_implementation == "flash_attention_2":
        pad_to_multiple_of = 2048
    else:
        pad_to_multiple_of = None

    inputs = get_model_inputs(
        orig_model,
        model_name_or_path,
        batch_size=dp_size,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    xla_inputs = {k: v.to(xm.xla_device()) for k, v in inputs.items()}
    xm.mark_step()

    with torch.no_grad():
        orig_model_outputs = orig_model(**xla_inputs)

    xm.mark_step()

    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        fuse_qkv=qkv_implementation == "fuse_qkv",
        recompute_causal_mask=False,  # Recomputing the causal mask does not impact the loss but it impacts the logits.
    )

    training_mod = importlib.import_module("optimum.neuron.models.training")
    custom_model_class = getattr(training_mod, model_class_name)
    with static_seed_patcher:
        model = custom_model_class.from_pretrained(
            model_name_or_path, trn_config, attn_implementation=attn_implementation, torch_dtype=torch_dtype
        )
        move_model_to_device(model, xm.xla_device())

    with static_seed_patcher:
        model = accelerator.prepare(model)

    xm.mark_step()

    with torch.no_grad():
        if pp_size == 1:
            # This is set to False by `accelerator.prepare`, which we want in the general case, but here let's
            # enable the cache to test that the KV cache matches the original model.
            model = model.eval()
            model_outputs = model(**xla_inputs)
        else:
            loss = model.run_eval(**inputs)
            model_outputs = {"loss": loss}

    xm.mark_step()

    outputs_to_consider = [output_name for output_name in model_outputs if output_name not in OUTPUTS_TO_IGNORE]

    if pp_size > 1:
        outputs_to_consider = ["loss"]

    outputs_to_check = [
        (orig_model_outputs[output_name], model_outputs[output_name]) for output_name in outputs_to_consider
    ]
    outputs_to_check = pytree.tree_map(move_all_tensor_to_cpu, outputs_to_check)

    for output_name, outputs in zip(outputs_to_consider, outputs_to_check):
        # For now ignoring past_key_values because they do not match and it is not needed for training.
        if "past" in output_name:
            continue
        if all(output is None for output in outputs):
            continue
        if pp_size == 1 or pp_rank == pp_size - 1:
            _check_output(output_name, outputs[0], outputs[1])


CUSTOM_MODELINGS_TO_TEST = [("LlamaForCausalLM", "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random")]


@pytest.mark.parametrize("qkv_implementation", ["regular_qkv", "fuse_qkv", "qkv_linear"])
# We only test for [world_size, tp_size, pp_size] = [32, 2, 4] e.g. dp=8,tp=2,pp=2
@pytest.mark.parametrize("world_size,tp_size,pp_size", [[32, 2, 4]], ids=["dp=8,tp=2,pp=4"])
@pytest.mark.parametrize("model_specs", CUSTOM_MODELINGS_TO_TEST, ids=[specs[0] for specs in CUSTOM_MODELINGS_TO_TEST])
def test_custom_modeling_matches_original(
    model_specs,
    qkv_implementation,
    world_size,
    tp_size,
    pp_size,
    monkeypatch,
    tmpdir,
):
    # We could make these parameters but we do not want to test all combinations.
    sequence_parallel_enabled = True
    # The best default to test would be flash attention since it's the most performant, but it seems to produce
    # different outputs and cannot handle padding (to validate).
    attn_implementation = "eager"

    tmpdir = Path(tmpdir)
    new_model_name_or_path = tmpdir / "my_custom_model"
    model_class_name, model_name_or_path = model_specs

    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if qkv_implementation == "fuse_qkv":
        config.num_key_value_heads = config.num_attention_heads
        model_class = getattr(transformers, model_class_name)
        model = model_class.from_pretrained(model_name_or_path, config=config, ignore_mismatched_sizes=True)
        model.save_pretrained(new_model_name_or_path)
        tokenizer.save_pretrained(new_model_name_or_path)
        model_name_or_path = new_model_name_or_path
    elif qkv_implementation == "qkv_linear":
        tp_size = 2 * config.num_key_value_heads

    run_fn = partial(
        _custom_model_matches_original_model,
        model_class_name,
        model_name_or_path,
        (world_size, tp_size, pp_size),
        sequence_parallel_enabled,
        qkv_implementation,
        attn_implementation,
        monkeypatch,
    )
    launch_procs(run_fn, world_size, tp_size, pp_size)
