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

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Type, Union

import pytest
import torch
import torch.utils._pytree as pytree
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_BACKBONE_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
)

import optimum
from optimum.neuron.accelerate.accelerator import NeuronAccelerator
from optimum.neuron.distributed.parallelizers_manager import ParallelizersManager
from optimum.neuron.utils.cache_utils import (
    get_num_neuron_cores,
)
from optimum.neuron.utils.import_utils import (
    is_neuronx_available,
    is_neuronx_distributed_available,
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import set_neuron_cc_optlevel_for_model

from .distributed import DistributedTest
from .utils import SEED, create_accelerator_for_mp, get_model, get_model_inputs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
    from neuronx_distributed.modules.qkv_linear import get_kv_shared_group
    from neuronx_distributed.parallel_layers.parallel_state import (
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
    "T5ForSequenceClassification",
]


def _generate_supported_model_classes(
    model_type: str,
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[Type["PreTrainedModel"]]:
    task_mapping = {
        # TODO: enable that when base models are supported.
        # "default": MODEL_MAPPING,
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING,
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING,
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
        "speech-seq2seq": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
        # Those architectures are more painful to deal with because the input is different.
        # "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
        "document-question-answering": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
        "zero-shot-image-classification": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
        "ctc": MODEL_FOR_CTC_MAPPING,
        "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
        "semantic-segmentation": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
        "backbone": MODEL_FOR_BACKBONE_MAPPING,
    }

    if supported_tasks is None:
        supported_tasks = list(task_mapping.keys())
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]

    model_classes = []
    for task in supported_tasks:
        config_class = CONFIG_MAPPING[model_type]
        model_class = task_mapping[task].get(config_class, None)
        if model_class is not None and model_class not in CLASSES_TO_IGNORE:
            model_classes.append(model_class)

    return list(set(model_classes))


MODEL_TYPES_TO_TEST = [
    ("bert", "hf-internal-testing/tiny-random-bert", {"num_hidden_layers": "2"}),
    ("roberta", "hf-internal-testing/tiny-random-roberta", {"num_hidden_layers": "2"}),
    (
        "gpt_neo",
        "hf-internal-testing/tiny-random-GPTNeoModel",
        {
            "num_layers": "2",
        },
    ),
    (
        "gpt_neox",
        "michaelbenayoun/gpt-neox-tiny-4layers-random",
        {"num_hidden_layers": "2"},
    ),
    (
        "llama",
        "michaelbenayoun/llama-2-tiny-16layers-random",
    ),
    (
        "t5",
        "hf-internal-testing/tiny-random-T5Model",
        {"d_ff": "36", "num_layers": "2", "num_decoder_layers": "2"},
    ),
    ("mistral", "michaelbenayoun/mistral-tiny-4layers-8kv-heads-random"),
]

MODELS_TO_TEST = []
for entry in MODEL_TYPES_TO_TEST:
    if len(entry) == 2:
        model_type, model_name_or_path = entry
        config_overwrite = None
    else:
        model_type, model_name_or_path, config_overwrite = entry
    for model_class in _generate_supported_model_classes(model_type):
        entry = (model_type, model_class, model_name_or_path, config_overwrite)
        if entry not in MODELS_TO_TEST:
            MODELS_TO_TEST.append(entry)


MODEL_CLASSES_TO_IGNORE = [
    "BertForPreTraining",  # There is a compilation issue, and testing TP for BertForPretraining is not really important.
    # TODO
    # GPTNeo's attention mechanism is broken in transformers==4.36.2, this should be re-enabled once there is a release
    # containing this PR: https://github.com/huggingface/transformers/pull/28533
    "GPTNeoForSequenceClassification",
    "GPTNeoForTokenClassification",
    "GPTNeoForQuestionAnswering",
    "GPTNeoForCausalLM",
]


LLAMA_GQA_VARIANTS_TO_TEST = {
    # "MHA-setup": (
    #     8,
    #     2,
    #     1,
    #     {
    #         "num_hidden_layers": "2",
    #         "num_attention_heads": "8",
    #         "num_key_value_heads": "8",
    #     },
    # ),
    # "num_key_value_heads > tp_size": (
    #     8,
    #     2,
    #     1,
    #     {
    #         "num_hidden_layers": "2",
    #         "num_attention_heads": "8",
    #         "num_key_value_heads": "4",
    #     },
    # ),
    # "num_key_value_heads = tp_size": (
    #     8,
    #     8,
    #     1,
    #     {
    #         "num_hidden_layers": "2",
    #         "hidden_size": "32",
    #         "num_attention_heads": "16",
    #         "num_key_value_heads": "8",
    #     },
    # ),
    "num_key_value_heads < tp_size": (
        8,
        8,
        1,
        {
            "num_hidden_layers": "2",
            "hidden_size": "32",
            "num_attention_heads": "16",
            "num_key_value_heads": "2",
        },
    ),
    "MQA-setup": (
        8,
        8,
        1,
        {
            "num_hidden_layers": "2",
            "hidden_size": "32",
            "num_attention_heads": "16",
            "num_key_value_heads": "1",
        },
    ),
}
# LLAMA_V2_MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-32kv-heads-random"
# LLAMA_V2_MODEL_NAME = "anushehchaudry/llama-2-tiny-random"
# LLAMA_V2_MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-random"
LLAMA_V2_MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-32kv-heads-random"


@is_trainium_test
class TestModelParallelization(DistributedTest):
    OUTPUTS_TO_IGNORE = {
        # It might not match in the sequence parallel setting because of mistmatched shapes.
        # Since these outputs are not needed during training, we do not want to perform an expensive gather for them.
        "encoder_last_hidden_state",
    }

    @pytest.fixture(scope="class", params=[[2, 2, 1], [2, 1, 2], [16, 2, 2]], ids=["tp=2", "pp=2", "dp=4,tp=pp=2"])
    def parallel_sizes(self, request):
        return request.param

    @pytest.fixture(scope="class", params=MODELS_TO_TEST, ids=[specs[1].__name__ for specs in MODELS_TO_TEST])
    def model_specs(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[True, False], ids=["from_pretrained", "from_config"])
    def from_pretrained(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["no_lazy_load", "lazy_load"])
    def lazy_load(self, request):
        return request.param

    @pytest.fixture(
        scope="class", params=[False, True], ids=["sequence_parallel_disabled", "sequence_parallel_enabled"]
    )
    def sequence_parallel_enabled(self, request):
        return request.param

    def early_skip(self, fixtures_kwargs):
        pp_size = fixtures_kwargs.get("pp_size", None)
        parallel_sizes = fixtures_kwargs.get("parallel_sizes", None)
        if pp_size is None and parallel_sizes is not None:
            pp_size = parallel_sizes[-1]
        model_specs = fixtures_kwargs.get("model_specs", None)

        if pp_size > 1 and model_specs is not None:
            model_type = model_specs[0]
            manager = ParallelizersManager.parallelizer_for_model(model_type)
            if not manager.supports_pipeline_parallelism():
                pytest.skip(f"Pipeline parallelism is not supported for {model_class.__name__}.")

        return super().early_skip(fixtures_kwargs)

    def _check_output(self, name: str, original_output, output):
        assert type(original_output) is type(output)
        if isinstance(original_output, (tuple, list, set)):
            for idx, orig_output in enumerate(original_output):
                new_name = f"{name}.{idx}"
                self._check_output(new_name, orig_output, output[idx])
        elif isinstance(original_output, dict):
            for output_name in original_output:
                new_name = f"{name}.{output_name}"
                self._check_output(new_name, original_output[name], output[name])
        elif isinstance(original_output, torch.Tensor):
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
                output = output[:, ::kv_size_multiplier]

            torch.testing.assert_close(original_output, output)
        else:
            assert original_output == output, f"Output named {name} do not match."

    def _parallel_model_matches_original_model(
        self,
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
        print(orig_model.model.layers[1].self_attn.v_proj.weight)
        orig_model = NeuronAccelerator.patch_model_for_neuron(orig_model)

        set_neuron_cc_optlevel_for_model(orig_model)

        move_model_to_device(orig_model, xm.xla_device())
        orig_model = orig_model.eval()

        manager = ParallelizersManager.parallelizer_for_model(orig_model)

        if pp_size > 1 and not manager.supports_pipeline_parallelism():
            pytest.skip(f"Pipeline parallelism is not supported for {model_class.__name__}.")

        if sequence_parallel_enabled and not manager.supports_sequence_parallelism():
            pytest.skip(f"Sequence parallelism is not supported for {model_class.__name__}.")

        pad_to_multiple_of = None if not sequence_parallel_enabled else tp_size
        inputs = get_model_inputs(
            orig_model, model_name_or_path, batch_size=dp_size, pad_to_multiple_of=pad_to_multiple_of
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

        accelerator = create_accelerator_for_mp(
            tp_size,
            pp_size,
            parallelize_embeddings=parallelize_embeddings,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

        from .utils import create_static_seed_patcher

        static_seed_patcher = create_static_seed_patcher(model.__class__, SEED)
        with static_seed_patcher:
            model = accelerator.prepare(model)

        xm.mark_step()

        model = accelerator.patch_model_for_neuron(model)
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

        outputs_to_consider = [
            output_name for output_name in orig_model_outputs if output_name not in self.OUTPUTS_TO_IGNORE
        ]

        if pp_size > 1:
            outputs_to_consider = ["loss"]

        outputs_to_check = [
            (orig_model_outputs[output_name], model_outputs[output_name]) for output_name in outputs_to_consider
        ]
        outputs_to_check = pytree.tree_map(move_all_tensor_to_cpu, outputs_to_check)

        for output_name, outputs in zip(outputs_to_consider, outputs_to_check):
            if all(output is None for output in outputs):
                continue
            if pp_size == 1 or pp_rank == pp_size - 1:
                self._check_output(output_name, outputs[0], outputs[1])

    def test_parallel_model_matches_original_model_from_pretrained_with_parallel_embeddings_and_sequence_parallel(
        self,
        model_specs,
        parallel_sizes,
        monkeypatch,
    ):
        _, model_class, model_name_or_path, config_overwrite = model_specs
        monkeypatch.setattr(
            optimum.neuron.distributed.parallel_layers, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True
        )
        return self._parallel_model_matches_original_model(
            model_class, model_name_or_path, config_overwrite, parallel_sizes, True, True, True, True
        )

    @pytest.mark.skip("Model parallelism from config is not fully supported yet.")
    def test_parallel_model_matches_original_model_from_config(
        self,
        model_specs,
        parallel_sizes,
        monkeypatch,
    ):
        _, model_class, model_name_or_path, config_overwrite = model_specs
        monkeypatch.setattr(
            optimum.neuron.distributed.parallel_layers, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True
        )
        return self._parallel_model_matches_original_model(
            model_class, model_name_or_path, config_overwrite, parallel_sizes, False, True, False, False
        )

    @pytest.mark.skipif(
        NUM_NEURON_CORES_AVAILABLE < 32,
        reason=f"This test requires 32 Neuron cores, but only {NUM_NEURON_CORES_AVAILABLE} are available",
    )
    @pytest.mark.parametrize(
        "world_size,tp_size,pp_size,config_overwrite",
        LLAMA_GQA_VARIANTS_TO_TEST.values(),
        ids=LLAMA_GQA_VARIANTS_TO_TEST.keys(),
    )
    def test_llama_v2_gqa_with_qkv_parallel_collumn_linear(
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
    ):
        monkeypatch.setattr(
            optimum.neuron.distributed.parallel_layers, "_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT", True
        )
        num_kv_heads = int(config_overwrite["num_key_value_heads"])
        if num_kv_heads >= tp_size and (from_pretrained or lazy_load or sequence_parallel_enabled):
            pytest.skip("No need to test this setting.")

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
            False,  # lazy_load, # lazy_load,
            False,  # sequence_parallel_enabled,
            False,
            # True,
        )
