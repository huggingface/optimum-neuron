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
import torch_xla.core.xla_model as xm
import transformers
from neuronx_distributed.parallel_layers.parallel_state import (
    get_kv_shared_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from neuronx_distributed.utils.model_utils import move_model_to_device
from transformers import AutoConfig, AutoTokenizer
from transformers import LlamaForCausalLM as OriginalLlamaForCausalLM

import optimum
import optimum.neuron.models.training
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.models.training.llama.modeling_llama import LlamaForCausalLM
from optimum.neuron.models.training.transformations_utils import GQAQKVColumnParallelLinearSpec
from optimum.neuron.utils.import_utils import (
    is_neuronx_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from ..distributed_utils import distributed_test, run_distributed_test
from .utils import SEED, StaticSeedPatcher, create_accelerator, get_model_inputs


if is_neuronx_available():
    pass


CUSTOM_MODELINGS_TO_TEST = [
    ("LlamaForCausalLM", "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"),
    ("GraniteForCausalLM", "michaelbenayoun/granite-tiny-4kv-heads-4layers-random"),
    ("Qwen3ForCausalLM", "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"),
]
LLAMA_V2_MODEL_NAME = "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"


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

        torch.testing.assert_close(output, original_output)
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

    monkeypatch.setattr(
        optimum.neuron.models.training.loss_utils, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True
    )

    static_seed_patcher = StaticSeedPatcher(SEED)

    accelerator = create_accelerator(tp_size, pp_size, sequence_parallel_enabled=sequence_parallel_enabled)

    orig_model_class = getattr(transformers, model_class_name)
    with static_seed_patcher:
        # We need to specify `attn_implementation="eager"` to ensure that the original model does not use
        # another default such as sdpa or flash attention.
        orig_model = orig_model_class.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, attn_implementation="eager"
        )

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

    training_mod = importlib.import_module("optimum.neuron.models.training")
    custom_model_class = getattr(training_mod, model_class_name)

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
@pytest.mark.parametrize("model_specs", CUSTOM_MODELINGS_TO_TEST, ids=[specs[0] for specs in CUSTOM_MODELINGS_TO_TEST])
@pytest.mark.flaky(reruns=5, reruns_delay=5)
@is_trainium_test
def test_custom_modeling_matches_original(
    model_specs,
    qkv_implementation,
    monkeypatch,
    tmpdir,
    set_cache_for_ci,  # This fixture will handle setting the remote cache to make this test faster.
):
    # We could make these parameters but we do not want to test all combinations.
    sequence_parallel_enabled = True
    # The best default to test would be flash attention since it's the most performant, but it seems to produce
    # different outputs and cannot handle padding (to validate).
    attn_implementation = "eager"

    tmpdir = Path(tmpdir)
    new_model_name_or_path = tmpdir / "my_custom_model"
    model_class_name, model_name_or_path = model_specs

    training_mod = importlib.import_module("optimum.neuron.models.training")
    custom_model_class = getattr(training_mod, model_class_name)
    if custom_model_class.supports_pipeline_parallelism():
        world_size = 32
        tp_size = 2
        pp_size = 4
    else:
        world_size = 32
        tp_size = 2
        pp_size = 1

    config = AutoConfig.from_pretrained(model_name_or_path)
    if pp_size == 1:
        if not config.tie_word_embeddings:
            # We test with `tie_word_embeddings=True` because it is more restrictive.
            config.tie_word_embeddings = True
    else:
        # `tie_word_embeddings` is not supported in the PP setting.
        config.tie_word_embeddings = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if qkv_implementation == "fuse_qkv":
        config.num_key_value_heads = config.num_attention_heads
    elif qkv_implementation == "qkv_linear":
        tp_size = 2 * config.num_key_value_heads

    # Saving a local version compatible for what we want to test.
    model_class = getattr(transformers, model_class_name)
    model = model_class.from_pretrained(model_name_or_path, config=config, ignore_mismatched_sizes=True)
    model.save_pretrained(new_model_name_or_path)
    tokenizer.save_pretrained(new_model_name_or_path)
    model_name_or_path = new_model_name_or_path

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
    run_distributed_test(run_fn, world_size=world_size, tp_size=tp_size, pp_size=pp_size)


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


@distributed_test(
    world_size=2,
    tp_size=2,
    pp_size=1,
)
@is_trainium_test
def test_custom_model_resize_embedding():
    tp_size = get_tensor_model_parallel_size()

    # Use a small model config for testing
    config = AutoConfig.from_pretrained(LLAMA_V2_MODEL_NAME)

    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        sequence_parallel_enabled=True,
    )

    static_seed_patcher = StaticSeedPatcher(42)

    with static_seed_patcher:
        # Create model with custom TP layers (ParallelEmbedding, ColumnParallelLinear)
        model = LlamaForCausalLM(config, trn_config)
        model.eval()

        # Test the new functionality
        old_vocab_size = model.config.vocab_size

        # Test 1: Should fail when new vocab size is not divisible by TP size
        if tp_size > 1:
            invalid_vocab_size = old_vocab_size + 1  # Not divisible by tp_size
            with pytest.raises(ValueError, match="must be divisible by tensor parallel size"):
                model.resize_token_embeddings(invalid_vocab_size)

        # Test 2: Should succeed when divisible by TP size
        new_vocab_size = old_vocab_size + tp_size  # Ensure divisible by TP size

        # Check initial embedding types
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()

        # These should be TP layers
        assert "ParallelEmbedding" in str(type(input_embeddings))
        assert "ColumnParallelLinear" in str(type(output_embeddings))

        # Store original shapes
        old_input_shape = input_embeddings.weight.shape
        old_output_shape = output_embeddings.weight.shape

        # Test resizing
        model.resize_token_embeddings(new_vocab_size, mean_resizing=False)

        # Check that resizing worked
        new_input_embeddings = model.get_input_embeddings()
        new_output_embeddings = model.get_output_embeddings()

        # Check vocab size was updated
        assert model.config.vocab_size == new_vocab_size

        # Check local shapes (per TP rank)
        expected_local_vocab = new_vocab_size // tp_size
        assert new_input_embeddings.weight.shape[0] == expected_local_vocab
        assert new_output_embeddings.weight.shape[0] == expected_local_vocab

        # Ensure embedding dim didn't change
        assert new_input_embeddings.weight.shape[1] == old_input_shape[1]
        assert new_output_embeddings.weight.shape[1] == old_output_shape[1]

        # Test that existing weights were preserved
        old_local_vocab = old_vocab_size // tp_size
        assert torch.allclose(
            new_input_embeddings.weight[:old_local_vocab, :], input_embeddings.weight[:old_local_vocab, :]
        )
        assert torch.allclose(
            new_output_embeddings.weight[:old_local_vocab, :], output_embeddings.weight[:old_local_vocab, :]
        )

        # Test that new tokens have been initialized (not zero)
        if expected_local_vocab > old_local_vocab:
            new_tokens_input = new_input_embeddings.weight[old_local_vocab:, :]
            new_tokens_output = new_output_embeddings.weight[old_local_vocab:, :]
            assert not torch.allclose(new_tokens_input, torch.zeros_like(new_tokens_input))
            assert not torch.allclose(new_tokens_output, torch.zeros_like(new_tokens_output))


@is_trainium_test
@distributed_test(world_size=2, tp_size=2, pp_size=1)
def test_custom_model_tie_weights(tmpdir, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    config = AutoConfig.from_pretrained(LLAMA_V2_MODEL_NAME)

    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        sequence_parallel_enabled=True,
    )

    static_seed_patcher = StaticSeedPatcher(42)

    # Test case 1: Weights should be tied from initialization when tie_word_embeddings=True
    config.tie_word_embeddings = True

    with static_seed_patcher:
        model_tied = LlamaForCausalLM(config, trn_config)
        model_tied.eval()

    input_embeddings = model_tied.get_input_embeddings()
    output_embeddings = model_tied.get_output_embeddings()

    # Should already be tied from initialization
    assert input_embeddings.weight.storage().data_ptr() == output_embeddings.weight.storage().data_ptr()

    # Should be still tied after moved to the XLA device
    accelerator = create_accelerator(tp_size, pp_size, sequence_parallel_enabled=True)
    model_tied = accelerator.prepare(model_tied)
    assert input_embeddings.weight is output_embeddings.weight
    assert input_embeddings.weight.device.type == "xla"

    # Test case 2: Weights should also be tied from from_pretrained
    # We save the model with `tie_word_embeddings=True` to ensure that the weights are tied in the checkpoint and that
    # there is no `lm_head` weight in the checkpoint.
    model_name_or_path = tmpdir / "model_with_tied_weights"
    orig_model = OriginalLlamaForCausalLM.from_pretrained(LLAMA_V2_MODEL_NAME, config=config)
    orig_model.save_pretrained(model_name_or_path)
    with static_seed_patcher:
        model_pretrained = LlamaForCausalLM.from_pretrained(model_name_or_path, trn_config, config=config)

    input_emb_pretrained = model_pretrained.get_input_embeddings()
    output_emb_pretrained = model_pretrained.get_output_embeddings()

    assert input_emb_pretrained.weight.storage().data_ptr() == output_emb_pretrained.weight.storage().data_ptr()

    # Test case 3: Verify modifications propagate in tied weights
    original_weight = input_embeddings.weight.data.clone()

    with torch.no_grad():
        input_embeddings.weight.data += 0.1

    torch.testing.assert_close(output_embeddings.weight, original_weight + 0.1)

    # Test case 4: tie_weights() should maintain tying
    model_tied.tie_weights()
    assert input_embeddings.weight.storage().data_ptr() == output_embeddings.weight.storage().data_ptr()

    # Test case 5: Untied behavior when tie_word_embeddings=False
    config.tie_word_embeddings = False

    with static_seed_patcher:
        model_untied = LlamaForCausalLM(config, trn_config)
        model_untied.eval()

    input_emb_untied = model_untied.get_input_embeddings()
    output_emb_untied = model_untied.get_output_embeddings()

    assert input_emb_untied.weight.storage().data_ptr() != output_emb_untied.weight.storage().data_ptr()

    # Test case 6: Weights should also not be tied from from_pretrained
    with static_seed_patcher:
        model_pretrained = LlamaForCausalLM.from_pretrained(LLAMA_V2_MODEL_NAME, trn_config, config=config)

    input_emb_untied = model_untied.get_input_embeddings()
    output_emb_untied = model_untied.get_output_embeddings()

    assert input_emb_untied.weight.storage().data_ptr() != output_emb_untied.weight.storage().data_ptr()


@distributed_test(world_size=32, tp_size=2, pp_size=4)
@is_trainium_test
def test_each_pp_rank_only_loads_relevant_parameters(set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()
    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    model = LlamaForCausalLM.from_pretrained(LLAMA_V2_MODEL_NAME, trn_config)
    parameters_on_cpu = {n for n, p in model.named_parameters() if p.device == torch.device("cpu")}
    parameters_on_meta = {n for n, p in model.named_parameters() if p.device == torch.device("meta")}

    accelerator = create_accelerator(tp_size, pp_size)

    nxd_pp_model = accelerator.prepare(model)

    local_parameters = {n for n, _ in nxd_pp_model.local_named_parameters()}
    other_parameters = {n for n, _ in nxd_pp_model.named_parameters() if n not in local_parameters}

    diff = local_parameters ^ parameters_on_cpu
    assert diff != {}, f"Expected that only the parameters of the current PP rank are on CPU. Got {diff} instead."

    diff = other_parameters ^ parameters_on_meta
    assert diff != {}, (
        f"Expected that the parameters of the other PP ranks are on the meta device. Got {diff} instead."
    )
