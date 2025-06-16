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
from optimum.neuron.models.training.transformations_utils import GQAQKVColumnParallelLinearSpec
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.models.training.granite.modeling_granite import GraniteForCausalLM
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
        get_kv_shared_group,
        get_pipeline_model_parallel_rank,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
    from neuronx_distributed.utils.model_utils import move_model_to_device

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_available():
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
    from neuronx_distributed.utils.model_utils import move_model_to_device


CUSTOM_MODELINGS_TO_TEST = [("LlamaForCausalLM", "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random")]


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


OUTPUTS_TO_IGNORE = {
    # It might not match in the sequence parallel setting because of mistmatched shapes.
    # Since these outputs are not needed during training, we do not want to perform an expensive gather for them.
    "encoder_last_hidden_state",
}


def _check_output(name: str, original_output, output):
    assert type(original_output) is type(output)
    if isinstance(original_output, (tuple, list, set)):
        for idx, orig_output in enumerate(original_output):
            new_name = f"{name}.{idx}"
            _check_output(new_name, orig_output, output[idx])
    elif isinstance(original_output, dict):
        for output_name in original_output:
            new_name = f"{name}.{output_name}"
            _check_output(new_name, original_output[name], output[name])
    elif isinstance(original_output, torch.Tensor):
        # For now the past key values do not match, we ignore that as it does not impact training.
        xm.master_print(f"Comparing output named {name}")
        tp_size = get_tensor_model_parallel_size()
        tp_group = get_tensor_model_parallel_group()
        if original_output.shape != output.shape:
            gather_dim = min(
                idx for idx in range(original_output.dim()) if original_output.shape[idx] != output.shape[idx]
            )
            output = output.to(xm.xla_device())
            gathered = [torch.empty_like(output) for _ in range(tp_size)]
            torch.distributed.all_gather(gathered, output, group=tp_group)
            gathered_output = torch.cat(gathered, dim=gather_dim)
            xm.mark_step()
            output = gathered_output.to("cpu")

        # In this case, we assume GQAQKVColumnParallelLinear was used, we retrieve only the non-repeated KV heads.
        if "past" in name and original_output.size(1) != output.size(1):
            kv_size_multiplier = len(get_kv_shared_group(as_list=True)[0])
            output = torch.chunk(output, kv_size_multiplier, dim=1)[0]

        torch.testing.assert_close(original_output, output)
    else:
        assert original_output == output, f"Output named {name} do not match."


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
    world_size, tp_size, pp_size = parallel_sizes
    dp_size = world_size // (tp_size * pp_size)
    pp_rank = get_pipeline_model_parallel_rank()

    training_mod = importlib.import_module("optimum.neuron.models.training")
    custom_model_class = getattr(training_mod, model_class_name)

    if pp_size > 1 and not custom_model_class.supports_pipeline_parallelism():
        pytest.skip(f"The model {model_class_name} does not support pipeline parallelism, skipping the test.")

    monkeypatch.setattr(
        optimum.neuron.models.training.loss_utils, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True
    )

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

    with static_seed_patcher:
        model = custom_model_class.from_pretrained(
            model_name_or_path, trn_config, attn_implementation=attn_implementation, torch_dtype=torch_dtype
        )

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


@pytest.mark.parametrize("qkv_implementation", ["regular_qkv", "fuse_qkv", "qkv_linear"])
# We only test for [world_size, tp_size, pp_size] = [32, 2, 4] e.g. dp=4,tp=2,pp=4
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

@pytest.mark.parametrize(
    "tp_size,num_attention_heads,num_key_value_heads,kv_size_multiplier,ground_truth",
    [
        [
            8,
            32,
            4,
            2,
            [
                [0, 1, 2, 3],
                [8, 9, 10, 11],
                [16, 17, 18, 19],
                [24, 25, 26, 27],
                [4, 5, 6, 7],
                [12, 13, 14, 15],
                [20, 21, 22, 23],
                [28, 29, 30, 31],
            ],
        ],
        [
            8,
            32,
            4,
            4,
            [
                [0, 1, 8, 9],
                [16, 17, 24, 25],
                [2, 3, 10, 11],
                [18, 19, 26, 27],
                [4, 5, 12, 13],
                [20, 21, 28, 29],
                [6, 7, 14, 15],
                [22, 23, 30, 31],
            ],
        ],
        [
            8,
            32,
            4,
            8,
            [
                [0, 8, 16, 24],
                [1, 9, 17, 25],
                [2, 10, 18, 26],
                [3, 11, 19, 27],
                [4, 12, 20, 28],
                [5, 13, 21, 29],
                [6, 14, 22, 30],
                [7, 15, 23, 31],
            ],
        ],
        [
            32,
            32,
            4,
            8,
            [
                [0],
                [8],
                [16],
                [24],
                [1],
                [9],
                [17],
                [25],
                [2],
                [10],
                [18],
                [26],
                [3],
                [11],
                [19],
                [27],
                [4],
                [12],
                [20],
                [28],
                [5],
                [13],
                [21],
                [29],
                [6],
                [14],
                [22],
                [30],
                [7],
                [15],
                [23],
                [31],
            ],
        ],
    ],
    ids=[
        "32-heads-4kv-heads-kv-mul-2,one kv head per rank",
        "32-heads-4kv-heads-kv-mul-4,multiple kv heads per rank",
        "32-heads-4kv-heads-kv-mul-8,all kv heads per rank",
        "tp=32,32-heads-4kv-heads-kv-mul-8,one query head per rank",
    ],
)
@is_trainium_test
def test_compute_query_indices_for_rank(
    tp_size, num_attention_heads, num_key_value_heads, kv_size_multiplier, ground_truth
):
    for tp_rank in range(tp_size):
        expected = torch.tensor(ground_truth[tp_rank])
        computed = GQAQKVColumnParallelLinearSpec.compute_query_indices_for_rank(
            tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier
        )
        print(f"TP rank = {tp_rank}")
        print(f"Expected {expected}")
        print(f"Computed {computed}")
        torch.testing.assert_close(expected, computed)


@is_trainium_test
def _test_resize_embedding(tie_embeddings):
    tp_size = get_tensor_model_parallel_size()
    tp_group = get_tensor_model_parallel_group()

    static_seed_patcher = StaticSeedPatcher(42)

    config = AutoConfig.from_pretrained(LLAMA_V2_MODEL_NAME)
    config.tie_word_embeddings = tie_embeddings

    with static_seed_patcher:
        orig_model = AutoModelForCausalLM.from_pretrained(LLAMA_V2_MODEL_NAME, config=config)
        orig_model.eval()
        vocab_size = orig_model.config.vocab_size
        new_vocab_size = vocab_size + tp_size

    with static_seed_patcher:
        orig_model.resize_token_embeddings(new_vocab_size)

    with static_seed_patcher:
        model = AutoModelForCausalLM.from_pretrained(LLAMA_V2_MODEL_NAME, config=config)
        model.eval()

    with static_seed_patcher:
        model.resize_token_embeddings(new_vocab_size)

    accelerator = create_accelerator(
        tp_size,
        1,
        parallelize_embeddings=True,
        sequence_parallel_enabled=True,
    )
    with static_seed_patcher:
        model = accelerator.prepare_model(model)

    # First we check that the embedding weights match
    gathered = [torch.empty_like(model.model.embed_tokens.weight) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered, model.model.embed_tokens.weight, group=tp_group)
    gathered_embedding = torch.cat(gathered, dim=0)
    xm.mark_step()
    torch.testing.assert_close(orig_model.model.embed_tokens.weight, gathered_embedding.to("cpu"))

    # Second we check that logits match
    tok = AutoTokenizer.from_pretrained(LLAMA_V2_MODEL_NAME)
    tok.pad_token = tok.eos_token
    inputs = tok("This is a test", max_length=24, padding="max_length", return_tensors="pt")
    inputs = {k: v.to("xla") for k, v in inputs.items()}
    orig_model = orig_model.to("xla")
    orig_logits = orig_model(**inputs).logits
    xm.mark_step()
    logits = model(**inputs).logits
    xm.mark_step()
    gathered = [torch.empty_like(logits) for _ in range(tp_size)]
    torch.distributed.all_gather(gathered, logits, group=tp_group)
    gathered_logits = torch.cat(gathered, dim=2)
    xm.mark_step()
    torch.testing.assert_close(orig_logits, gathered_logits)


@is_trainium_test
@pytest.mark.parametrize(
    "tie_embeddings",
    [False, True],
    ids=["embeddings_not_tied", "tied_embeddings"],
)
def test_resize_embedding(tie_embeddings):
    world_size, tp_size, pp_size = (2, 2, 1)
    run_fn = partial(_test_resize_embedding, tie_embeddings)
    launch_procs(run_fn, world_size, tp_size, pp_size)
