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
"""Tests validating that models can be parallelized correctly."""

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Type, Union

import pytest
import torch
import torch.utils._pytree as pytree
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
)

import optimum
from optimum.neuron.distributed.parallelizers_manager import ParallelizersManager
from optimum.neuron.distributed.utils import compute_query_indices_for_rank, lazy_load_for_parallelism
from optimum.neuron.utils.cache_utils import (
    get_num_neuron_cores,
)
from optimum.neuron.utils.import_utils import (
    is_neuronx_available,
    is_neuronx_distributed_available,
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import DistributedTest, launch_procs
from ..utils import SEED, StaticSeedPatcher, create_accelerator, get_model, get_model_inputs


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

if TYPE_CHECKING:
    from transformers import PreTrainedModel


TEMPLATE_FILE_NAME = "model_parallel_test_template.txt"
if is_neuronx_available():
    NUM_NEURON_CORES_AVAILABLE = get_num_neuron_cores()
else:
    NUM_NEURON_CORES_AVAILABLE = 0


CLASSES_TO_IGNORE = [
    # TODO: enable this class when it can be traced for pipeline parallelism.
    "LlamaForQuestionAnswering",
]


def _generate_supported_model_classes(
    model_type: str,
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[Type["PreTrainedModel"]]:
    task_mapping = {
        # TODO: enable that when base models are supported.
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    }

    if supported_tasks is None:
        supported_tasks = list(task_mapping.keys())
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]

    model_classes = []
    for task in supported_tasks:
        config_class = CONFIG_MAPPING[model_type]
        model_class = task_mapping[task].get(config_class, None)
        if model_class is not None and model_class.__name__ not in CLASSES_TO_IGNORE:
            model_classes.append(model_class)

    return list(set(model_classes))


LLAMA_TYPES_TO_TEST = ("llama", "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random", None)
MODEL_TYPES_TO_TEST = [
    ("roberta", "hf-internal-testing/tiny-random-roberta", {"num_hidden_layers": "2"}),
    (
        "gpt_neo",
        "hf-internal-testing/tiny-random-GPTNeoModel",
        {
            "num_layers": "2",
        },
    ),
    (
        "t5",
        "hf-internal-testing/tiny-random-T5Model",
        {"d_ff": "36", "num_layers": "2", "num_decoder_layers": "2"},
    ),
    ("mistral", "michaelbenayoun/mistral-tiny-4layers-8kv-heads-random", None),
]


def _build_models_to_test(model_types_to_test):
    models_to_test = []
    for entry in model_types_to_test:
        model_type, model_name_or_path, config_overwrite = entry
        for model_class in _generate_supported_model_classes(model_type):
            models_to_test.append((model_type, model_class, model_name_or_path, config_overwrite))
    return models_to_test


LLAMA_MODELS_TO_TEST = _build_models_to_test([LLAMA_TYPES_TO_TEST])
NOT_LLAMA_TO_TEST = _build_models_to_test(MODEL_TYPES_TO_TEST)
MODELS_TO_TEST = NOT_LLAMA_TO_TEST + LLAMA_MODELS_TO_TEST


MODEL_CLASSES_TO_IGNORE = [
    "GPTNeoForSequenceClassification",
    "GPTNeoForTokenClassification",
    "GPTNeoForQuestionAnswering",
    "GPTNeoForCausalLM",
]


OUTPUTS_TO_IGNORE = {
    # It might not match in the sequence parallel setting because of mistmatched shapes.
    # Since these outputs are not needed during training, we do not want to perform an expensive gather for them.
    "encoder_last_hidden_state",
}


LLAMA_V2_MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-32kv-heads-random"


def _early_skip(pp_size=None, parallel_sizes=None, model_specs=None):
    if pp_size is None and parallel_sizes is not None:
        pp_size = parallel_sizes[-1]

    if pp_size is not None and pp_size > 1 and model_specs is not None:
        model_type = model_specs[0]
        manager = ParallelizersManager.parallelizer_for_model(model_type)
        if not manager.supports_pipeline_parallelism():
            pytest.skip(f"Pipeline parallelism is not supported for {model_specs[1].__name__}.")


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

        xm.master_print("Diff tensor:", original_output - output)
        torch.testing.assert_close(original_output, output)
    else:
        assert original_output == output, f"Output named {name} do not match."


def _parallel_model_matches_original_model(
    model_class,
    model_name_or_path,
    config_overwrite,
    parallel_sizes,
    from_pretrained,
    lazy_load,
    sequence_parallel_enabled,
    parallelize_embeddings,
):
    if model_class.__name__ in MODEL_CLASSES_TO_IGNORE:
        pytest.skip(f"Skipping test for {model_class.__name__} since it is buggy or a special case.")

    world_size, tp_size, pp_size = parallel_sizes
    dp_size = world_size // (tp_size * pp_size)
    pp_rank = get_pipeline_model_parallel_rank()

    orig_model = get_model(
        model_class,
        model_name_or_path,
        from_config=not from_pretrained,
        config_overwrite=config_overwrite,
        use_static_seed_patcher=True,
    )

    accelerator = create_accelerator(
        tp_size,
        pp_size,
        parallelize_embeddings=parallelize_embeddings,
        sequence_parallel_enabled=sequence_parallel_enabled,
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

    manager = ParallelizersManager.parallelizer_for_model(orig_model)

    if pp_size > 1 and not manager.supports_pipeline_parallelism():
        pytest.skip(f"Pipeline parallelism is not supported for {model_class.__name__}.")

    if sequence_parallel_enabled and not manager.supports_sequence_parallelism():
        pytest.skip(f"Sequence parallelism is not supported for {model_class.__name__}.")

    if not from_pretrained and lazy_load:
        pytest.skip("This is not supported, issue with tying weights.")

    pad_to_multiple_of = None if not sequence_parallel_enabled else tp_size
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

    # The parallel model needs to be defined after the forward pass of the first model because there is a
    # global monkey patching of the `torch.nn.CrossEntropyLoss` class when doing sequence parallelism.
    model = get_model(
        model_class,
        model_name_or_path,
        tp_size=tp_size,
        pp_size=pp_size,
        lazy_load=lazy_load,
        from_config=not from_pretrained,
        config_overwrite=config_overwrite,
        use_static_seed_patcher=True,
    )

    static_seed_patcher = StaticSeedPatcher(SEED)
    with static_seed_patcher:
        model = accelerator.prepare(model)

    xm.mark_step()

    with torch.no_grad():
        if pp_size == 1:
            # This is set to False by `accelerator.prepare`, which we want in the general case, but here let's
            # enable the cache to test that the KV cache matches the original model.
            model.config.use_cache = True
            model = model.eval()
            model_outputs = model(**xla_inputs)
        else:
            loss = model.run_eval(**inputs)
            model_outputs = {"loss": loss}

    xm.mark_step()

    outputs_to_consider = [output_name for output_name in orig_model_outputs if output_name not in OUTPUTS_TO_IGNORE]

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


@is_trainium_test
class TestModelParallelization(DistributedTest):
    @pytest.fixture(scope="class", params=[[2, 2, 1], [2, 1, 2], [16, 2, 2]], ids=["tp=2", "pp=2", "dp=4,tp=pp=2"])
    def parallel_sizes(self, request):
        return request.param

    @pytest.fixture(scope="class", params=MODELS_TO_TEST, ids=[specs[1].__name__ for specs in MODELS_TO_TEST])
    def model_specs(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[True, False], ids=["from_pretrained", "from_config"])
    def from_pretrained(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["regular_load", "lazy_load"])
    def lazy_load(self, request):
        return request.param

    @pytest.fixture(
        scope="class", params=[False, True], ids=["sequence_parallel_disabled", "sequence_parallel_enabled"]
    )
    def sequence_parallel_enabled(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["embeddings_not_parallel", "parallelized_embeddings"])
    def parallelize_embeddings(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["embeddings_not_tied", "tied_embeddings"])
    def tie_embeddings(self, request):
        return request.param

    @pytest.mark.skip("Model parallelism from config is not fully supported yet.")
    def test_parallel_model_matches_original_model_from_config(
        self,
        model_specs,
        parallel_sizes,
        monkeypatch,
    ):
        _, model_class, model_name_or_path, config_overwrite = model_specs
        monkeypatch.setattr(optimum.neuron.distributed.utils, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True)
        return _parallel_model_matches_original_model(
            model_class, model_name_or_path, config_overwrite, parallel_sizes, False, True, False, False
        )

    @pytest.mark.skipif(
        NUM_NEURON_CORES_AVAILABLE < 32,
        reason=f"This test requires 32 Neuron cores, but only {NUM_NEURON_CORES_AVAILABLE} are available",
    )
    @pytest.mark.parametrize(
        "world_size,tp_size,pp_size,config_overwrite",
        [
            [
                8,
                2,
                1,
                {
                    "num_hidden_layers": "2",
                    "hidden_size": "32",
                    "num_attention_heads": "8",
                    "num_key_value_heads": "8",
                },
            ],
            [
                8,
                2,
                1,
                {
                    "num_hidden_layers": "2",
                    "hidden_size": "32",
                    "num_attention_heads": "8",
                    "num_key_value_heads": "4",
                },
            ],
            [
                8,
                8,
                1,
                {
                    "num_hidden_layers": "2",
                    "hidden_size": "32",
                    "num_attention_heads": "16",
                    "num_key_value_heads": "8",
                },
            ],
            [
                8,
                8,
                1,
                {
                    "num_hidden_layers": "2",
                    "hidden_size": "32",
                    "num_attention_heads": "16",
                    "num_key_value_heads": "2",
                },
            ],
            [
                16,
                8,
                2,
                {
                    "num_hidden_layers": "2",
                    "hidden_size": "32",
                    "num_attention_heads": "16",
                    "num_key_value_heads": "2",
                },
            ],
            [
                8,
                8,
                1,
                {
                    "num_hidden_layers": "2",
                    "hidden_size": "32",
                    "num_attention_heads": "16",
                    "num_key_value_heads": "1",
                },
            ],
        ],
        ids=[
            "MHA-setup",
            "num_key_value_heads bigger than tp_size",
            "num_key_value_heads equal to tp_size",
            "num_key_value_heads lower than tp_size",
            "num_key_value_heads lower than tp_size,pp enabled",
            "MQA-setup",
        ],
    )
    def test_llama_v2_gqa(
        self,
        monkeypatch,
        tmpdir,
        world_size,
        tp_size,
        pp_size,
        config_overwrite,
        from_pretrained,
        lazy_load,
        sequence_parallel_enabled,
        parallelize_embeddings,
    ):
        monkeypatch.setattr(optimum.neuron.distributed.utils, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True)
        num_kv_heads = int(config_overwrite["num_key_value_heads"])
        # if num_kv_heads >= tp_size and (from_pretrained or lazy_load or sequence_parallel_enabled):
        #     pytest.skip("No need to test this setting.")

        # The following case can be skipped because since we set the seed, we would need to shuffle the output
        # projections for this case to work. This is not needed in the real-case scenario, and since we test for every
        # other setting, we can skip.
        if num_kv_heads < tp_size and (not from_pretrained):
            pytest.skip("This case will  not work here because we set the seed. We can skip.")

        model_name_or_path = Path(tmpdir) / "llama_v2_gqa"

        # Since we are creating the model from config, we actually first create a model locally from config and then
        # use that as a `from_pretrained` to have proper initialization. Without that we can end-up with uninitialized
        # weights.
        if xm.get_ordinal() == 0:
            tokenizer = AutoTokenizer.from_pretrained(LLAMA_V2_MODEL_NAME)
            tokenizer.save_pretrained(model_name_or_path)
            model = get_model(
                LlamaForCausalLM,
                LLAMA_V2_MODEL_NAME,
                from_config=True,
                config_overwrite=config_overwrite,
            )
            model.save_pretrained(model_name_or_path)
        xm.rendezvous("Model creation done.")

        return self._parallel_model_matches_original_model(
            LlamaForCausalLM,
            model_name_or_path,
            config_overwrite,
            (world_size, tp_size, pp_size),
            from_pretrained,
            lazy_load,
            sequence_parallel_enabled,
            parallelize_embeddings,
        )

    @pytest.mark.parallel_sizes((2, 2, 1))
    def test_resize_embedding(self, tie_embeddings, lazy_load):
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

        ctx = lazy_load_for_parallelism(tensor_parallel_size=tp_size) if lazy_load else nullcontext()
        with ctx:
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


def _test_parallelized_layers_model_matches_original(
    model_specs,
    world_size,
    tp_size,
    pp_size,
    monkeypatch,
):
    _, model_class, model_name_or_path, config_overwrite = model_specs

    # This is very important otherwise the parallel cross entropy loss will modify the logits inplace.
    monkeypatch.setattr(optimum.neuron.distributed.utils, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True)

    # Skip on certain combinations
    _early_skip(pp_size=pp_size, parallel_sizes=[world_size, tp_size, pp_size], model_specs=model_specs)
    parallel_sizes = world_size, tp_size, pp_size

    run_fn = partial(
        _parallel_model_matches_original_model,
        model_class,
        model_name_or_path,
        config_overwrite,
        parallel_sizes,
        True,
        True,
        True,
        True,
    )
    launch_procs(run_fn, world_size, tp_size, pp_size)


@is_trainium_test
@pytest.mark.parametrize(
    "world_size,tp_size,pp_size", [[2, 2, 1], [2, 1, 2], [16, 2, 2]], ids=["tp=2", "pp=2", "dp=4,tp=pp=2"]
)
@pytest.mark.parametrize("model_specs", NOT_LLAMA_TO_TEST, ids=[specs[1].__name__ for specs in NOT_LLAMA_TO_TEST])
def test_parallelized_layers_model_matches_original(
    model_specs,
    world_size,
    tp_size,
    pp_size,
    monkeypatch,
):
    return _test_parallelized_layers_model_matches_original(
        model_specs,
        world_size,
        tp_size,
        pp_size,
        monkeypatch,
    )


@is_trainium_test
@pytest.mark.parametrize(
    "world_size,tp_size,pp_size", [[2, 2, 1], [2, 1, 2], [16, 2, 2]], ids=["tp=2", "pp=2", "dp=4,tp=pp=2"]
)
@pytest.mark.parametrize(
    "model_specs", LLAMA_MODELS_TO_TEST, ids=[specs[1].__name__ for specs in LLAMA_MODELS_TO_TEST]
)
def test_parallelized_llama_matches_original(
    model_specs,
    world_size,
    tp_size,
    pp_size,
    monkeypatch,
):
    return _test_parallelized_layers_model_matches_original(
        model_specs,
        world_size,
        tp_size,
        pp_size,
        monkeypatch,
    )


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
        computed = compute_query_indices_for_rank(
            tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier
        )
        print(f"TP rank = {tp_rank}")
        print(f"Expected {expected}")
        print(f"Computed {computed}")
        torch.testing.assert_close(expected, computed)
