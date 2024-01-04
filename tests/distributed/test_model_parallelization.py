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

from typing import TYPE_CHECKING, List, Optional, Type, Union

import pytest
import torch
import torch.utils._pytree as pytree
from transformers import LlamaForCausalLM
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

from .distributed import DistributedTest
from .utils import create_accelerator_for_mp, get_model, get_model_inputs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
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
        "hf-tiny-model-private/tiny-random-GPTNeoXModel",
        {"num_hidden_layers": "2", "intermediate_size": "36"},
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


# When doing from pretrained + lazy loading, it is not always easy to initiliazed the remaining weights in a similar
# fashion than in the regular model. So we do not check for them under this specific setting. It does not mean that
# parallelization does not work for them, only that some weights cannot be initialized exactly the same way.
MODEL_CLASSES_TO_IGNORE_ON_LAZY_LOAD_FOR_FROM_PRETRAINED = [
    "T5ForQuestionAnswering",
]


LLAMA_GQA_VARIANTS_TO_TEST = {
    "MHA-setup": (
        8,
        2,
        1,
        {
            "num_hidden_layers": "2",
            "num_attention_heads": "8",
            "num_key_value_heads": "8",
        },
    ),
    "num_key_value_heads > tp_size": (
        8,
        2,
        1,
        {
            "num_hidden_layers": "2",
            "num_attention_heads": "8",
            "num_key_value_heads": "4",
        },
    ),
    "num_key_value_heads = tp_size": (
        8,
        8,
        1,
        {
            "num_hidden_layers": "2",
            "hidden_size": "32",
            "num_attention_heads": "16",
            "num_key_value_heads": "8",
        },
    ),
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
LLAMA_V2_MODEL_NAME = "anushehchaudry/llama-2-tiny-random"


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
            if original_output.shape != output.shape:
                gather_dim = min(
                    idx for idx in range(original_output.dim()) if original_output.shape[idx] != output.shape[idx]
                )
                output = output.to(xm.xla_device())
                gathered = [torch.empty_like(output) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered, output, group=get_tensor_model_parallel_group())
                gathered_output = torch.cat(gathered, dim=gather_dim)
                xm.mark_step()
                output = gathered_output.to("cpu")
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
        _, tp_size, pp_size = parallel_sizes
        pp_rank = get_pipeline_model_parallel_rank()

        orig_model = get_model(
            model_class,
            model_name_or_path,
            from_config=not from_pretrained,
            config_overwrite=config_overwrite,
            use_static_seed_patcher=True,
        )
        move_model_to_device(orig_model, xm.xla_device())
        orig_model = orig_model.eval()

        manager = ParallelizersManager.parallelizer_for_model(orig_model)

        if pp_size > 1 and not manager.supports_pipeline_parallelism():
            pytest.skip(f"Pipeline parallelism is not supported for {model_class.__name__}.")

        if sequence_parallel_enabled and not manager.supports_sequence_parallelism():
            pytest.skip(f"Sequence parallelism is not supported for {model_class.__name__}.")

        pad_to_multiple_of = None if not sequence_parallel_enabled else tp_size
        inputs = get_model_inputs(orig_model, model_name_or_path, pad_to_multiple_of=pad_to_multiple_of)

        xla_inputs = {k: v.to(xm.xla_device()) for k, v in inputs.items()}
        xm.mark_step()

        with torch.no_grad():
            orig_model_outputs = orig_model(**xla_inputs)

        xm.mark_step()

        # The parallel model needs to be define after the forward pass of the first model because there is a
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

        static_seed_patcher = create_static_seed_patcher(model.__class__, 42)
        with static_seed_patcher:
            model = accelerator.prepare(model)
        if xm.get_ordinal() == 0:
            pass
            # print(model.gpt_neox.embed_in.weight, orig_model.gpt_neox.embed_in.weight)
            # print(model.embed_out.weight, orig_model.embed_out.weight)
            # print(model.gpt_neox.embed_in.weight, model.embed_out.weight)

        with torch.no_grad():
            if pp_size == 1:
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
    def test_llama_v2_gqa_variants(self, world_size, tp_size, pp_size, config_overwrite):
        return self._parallel_model_matches_original_model(
            LlamaForCausalLM,
            LLAMA_V2_MODEL_NAME,
            config_overwrite,
            (world_size, tp_size, pp_size),
            False,
            False,
            False,
            False,
        )

    # def _test_model_parallel(
    #     self,
    #     tp_size: int,
    #     pp_size: int,
    #     model_class_name: str,
    #     model_name_or_path: str,
    #     from_config: bool,
    #     with_lazy_load: bool,
    #     parallelize_embeddings: bool,
    #     sequence_parallel_enabled: bool,
    #     num_neuron_cores: int = NUM_NEURON_CORES_AVAILABLE,
    #     run_test_in_parallel: bool = False,
    #     overwrite_model_config: Optional[Dict[str, str]] = None,
    # ):
    #     if "GPTNeoX" in model_class_name:
    #         self.skipTest("GPTNeoX test is flaky, needs to be fixed.")

    #     if num_neuron_cores < tp_size:
    #         raise ValueError(
    #             "The number of Neuron cores available is lower than the TP size, failing since the test might not be "
    #             "testing what is expected."
    #         )

    #     if run_test_in_parallel and (NUM_NEURON_CORES_AVAILABLE // num_neuron_cores) < 2:
    #         raise ValueError(
    #             "The test cannot be run in parallel because there is not enough Neuron cores available to preserve the "
    #             f"number of Neuron cores requested ({NUM_NEURON_CORES_AVAILABLE} cores available and {num_neuron_cores} "
    #             "were requested)"
    #         )

    #     template_content = None
    #     current_directory = Path(__file__).parent.resolve()
    #     template_file_path = current_directory / TEMPLATE_FILE_NAME
    #     with open(template_file_path, "r") as fp:
    #         template_content = fp.read()

    #     specialization_env = {
    #         "from_config": "true" if from_config else "false",
    #         "lazy_load": "true" if with_lazy_load else "false",
    #         "parallelize_embeddings": "true" if parallelize_embeddings else "false",
    #         "sequence_parallel_enabled": "true" if sequence_parallel_enabled else "false",
    #         "computing_loss_is_supported": "true",
    #         **os.environ,
    #     }

    #     # Updating the Python path to be able to use `tests/distributed/utils.py`.
    #     python_path = specialization_env.get("PYTHONPATH", "")
    #     python_path = f"{current_directory}:{python_path}"
    #     specialization_env["PYTHONPATH"] = python_path

    #     if overwrite_model_config is not None:
    #         specialization_env["config_overwrite"] = ",".join(
    #             f"{key}={value}" for key, value in overwrite_model_config.items()
    #         )

    #     with TemporaryDirectory() as tmpdirname:
    #         specialization_data = {
    #             "model_class": model_class_name,
    #             "model_name_or_path": model_name_or_path,
    #             "parallelize_embeddings": "True" if parallelize_embeddings else "False",
    #             "tp_size": tp_size,
    #             "pp_size": pp_size,
    #             "output_path": tmpdirname,
    #         }
    #         specialized_content = template_content.format(**specialization_data)
    #         with open(f"{tmpdirname}/code.py", "w") as fp:
    #             fp.write(specialized_content)

    #         cmd = ["torchrun", f"--nproc_per_node={num_neuron_cores}", f"{tmpdirname}/code.py"]

    #         # When running the test in parallel, we need 2 rendez-vous endpoints: one for the script running the
    #         # original model and one for the script running the parallel model.
    #         rdzv_endpoint_host = "localhost"
    #         rdzv_endpoint_port = 29400

    #         orig_neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    #         set_neuron_cache_path(tmpdirname)
    #         neuron_cc_flags = os.environ["NEURON_CC_FLAGS"]
    #         os.environ["NEURON_CC_FLAGS"] = orig_neuron_cc_flags

    #         # Original model.
    #         env = {"is_parallel": "false", **specialization_env, "NEURON_CC_FLAGS": neuron_cc_flags}
    #         if run_test_in_parallel:
    #             # Setting the rendez-vous endpoint for the original model process.
    #             cmd.insert(1, f"--rdzv_endpoint={rdzv_endpoint_host}:{rdzv_endpoint_port}")
    #             env["NEURON_RT_VISIBLE_CORES"] = f"0-{num_neuron_cores - 1}"

    #         # When running tests in parallel, synchronization is done after both processes started.
    #         if not run_test_in_parallel:
    #             p_original_returncode, stdout = run_command_with_realtime_output(cmd, env=env)
    #         else:
    #             p_original = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    #         # Parallel model.
    #         env = {"is_parallel": "true", **specialization_env, "NEURON_CC_FLAGS": neuron_cc_flags}
    #         if run_test_in_parallel:
    #             # Updating the rendez-vous endpoint for the parallel model process.
    #             cmd[1] = f"--rdzv_endpoint={rdzv_endpoint_host}:{rdzv_endpoint_port + 1}"
    #             env["NEURON_RT_VISIBLE_CORES"] = f"{num_neuron_cores}-{2 * num_neuron_cores - 1}"

    #             p_parallel = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    #             stdout, _ = p_original.communicate()
    #             p_original_returncode = p_original.returncode
    #             stdout = stdout.decode("utf-8")
    #             full_output = f"Original model standard output:\n{stdout}"
    #             print(full_output)

    #             stdout, _ = p_parallel.communicate()
    #             p_parallel_returncode = p_parallel.returncode
    #             stdout = stdout.decode("utf-8")
    #             full_output = f"Parallel model standard output:\n{stdout}"
    #             print(full_output)

    #         else:
    #             p_parallel_returncode, stdout = run_command_with_realtime_output(cmd, env=env)

    #         assert p_original_returncode == 0
    #         assert p_parallel_returncode == 0

    #         temporary_dir = Path(tmpdirname)
    #         original_model_outputs = torch.load(temporary_dir / "original.bin")
    #         parallel_model_outputs = torch.load(temporary_dir / "parallel.bin")

    #         if (
    #             not from_config
    #             and with_lazy_load
    #             and model_class_name in MODEL_CLASSES_TO_IGNORE_ON_LAZY_LOAD_FOR_FROM_PRETRAINED
    #         ):
    #             self.skipTest(
    #                 f"Cannot compare outputs for {model_class_name} when doing from_pretrained + lazy loading."
    #             )

    #         for name, t in original_model_outputs.items():
    #             if name in self.OUTPUTS_TO_IGNORE:
    #                 continue
    #             print(f"Testing that {name} match.")
    #             regular_parallel_outputs_error_msg = None
    #             gathered_parallel_outputs_error_msg = None
    #             try:
    #                 self._check_output(name, t, parallel_model_outputs[name], with_lazy_load)
    #             except AssertionError as e:
    #                 regular_parallel_outputs_error_msg = str(e)
    #             if regular_parallel_outputs_error_msg is not None:
    #                 print("Regular output did not match, testing with the gathered output...")
    #                 try:
    #                     self._check_output(name, t, parallel_model_outputs[f"gathered_{name}"], with_lazy_load)
    #                 except AssertionError as e:
    #                     gathered_parallel_outputs_error_msg = str(e)
    #             if regular_parallel_outputs_error_msg is not None and gathered_parallel_outputs_error_msg is not None:
    #                 msg = (
    #                     "Output did not matched.\nTest with non-gathered parallel outputs error:\n"
    #                     f"{regular_parallel_outputs_error_msg}\nTest with gathered parallel outputs error:\n"
    #                     f"{gathered_parallel_outputs_error_msg}"
    #                 )
    #                 raise AssertionError(msg)
    #             print("Ok!")

    # @parameterized.expand(MODELS_TO_TEST)
    # def test_model_parallel_from_config_no_lazy_load(
    #     self,
    #     model_type: str,
    #     model_class_name: str,
    #     model_name_or_path: str,
    #     config_overwrite: Dict[str, str],
    # ):
    #     # In this test, we:
    #     #   1. Test parallelism when initializing from a config.
    #     #   2. Do not enable embedding parallelization => while behaviour could differ between a model initialized
    #     #      lazily or not, the risk is minimal. This feature is tested on the next test with lazy loading.
    #     #   3. Do not enable sequence parallelism => this feature should not depend on whether the model is initialized
    #     #      lazily or not.
    #     def test_fn(tp_size: int, pp_size: int):
    #         self._test_model_parallel(
    #             tp_size=tp_size,
    #             pp_size=pp_size,
    #             num_neuron_cores=8,
    #             run_test_in_parallel=True,
    #             model_class_name=model_class_name,
    #             model_name_or_path=model_name_or_path,
    #             from_config=True,
    #             with_lazy_load=False,
    #             parallelize_embeddings=False,
    #             sequence_parallel_enabled=False,
    #             overwrite_model_config=config_overwrite,
    #         )

    #     with self.subTest("Test TP only"):
    #         tp_size = 2
    #         pp_size = 1
    #         test_fn(tp_size, pp_size)

    #     is_pp_supported = ParallelizersManager.parallelizer_for_model(model_type).supports_pipeline_parallelism()
    #     if is_pp_supported:
    #         with self.subTest("Test PP only"):
    #             tp_size = 1
    #             pp_size = 2
    #             test_fn(tp_size, pp_size)

    #         with self.subTest("Test TP + PP only"):
    #             tp_size = 2
    #             pp_size = 4
    #             test_fn(tp_size, pp_size)

    # @parameterized.expand(MODELS_TO_TEST)
    # def test_model_parallel_from_config_lazy_load(
    #     self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    # ):
    #     # In this test, we:
    #     #   1. Test parallelism when initializing lazily from a config.
    #     #   2. Enable embedding parallelization.
    #     #   3. Enable sequence parallelism.
    #     def test_fn(tp_size: int, pp_size: int):
    #         self._test_model_parallel(
    #             tp_size=tp_size,
    #             pp_size=pp_size,
    #             num_neuron_cores=8,
    #             run_test_in_parallel=True,
    #             model_class_name=model_class_name,
    #             model_name_or_path=model_name_or_path,
    #             from_config=True,
    #             with_lazy_load=True,
    #             parallelize_embeddings=True,
    #             sequence_parallel_enabled=True,
    #             overwrite_model_config=config_overwrite,
    #         )

    #     with self.subTest("Test TP only"):
    #         tp_size = 2
    #         pp_size = 1
    #         test_fn(tp_size, pp_size)

    #     is_pp_supported = ParallelizersManager.parallelizer_for_model(model_type).supports_pipeline_parallelism()
    #     if is_pp_supported:
    #         with self.subTest("Test PP only"):
    #             tp_size = 1
    #             pp_size = 2
    #             test_fn(tp_size, pp_size)

    #         with self.subTest("Test TP + PP only"):
    #             tp_size = 2
    #             pp_size = 4
    #             test_fn(tp_size, pp_size)

    # @parameterized.expand(MODELS_TO_TEST)
    # def test_model_parallel_from_pretrained_no_lazy_load(
    #     self,
    #     model_type: str,
    #     model_class_name: str,
    #     model_name_or_path: str,
    #     config_overwrite: Dict[str, str],
    # ):
    #     # In this test, we:
    #     #   1. Test parallelism when initializing from pretrained weights.
    #     #   2. Do not enable embedding parallelization => while behaviour could differ between a model initialized
    #     #      lazily or not, the risk is minimal. This feature is tested on the next test with lazy loading.
    #     #   3. Do not enable sequence parallelism => this feature should not depend on whether the model is initialized
    #     #      lazily or not.
    #     def test_fn(tp_size: int, pp_size: int):
    #         self._test_model_parallel(
    #             tp_size=tp_size,
    #             pp_size=pp_size,
    #             num_neuron_cores=8,
    #             run_test_in_parallel=True,
    #             model_class_name=model_class_name,
    #             model_name_or_path=model_name_or_path,
    #             from_config=False,
    #             with_lazy_load=False,
    #             parallelize_embeddings=False,
    #             sequence_parallel_enabled=False,
    #             overwrite_model_config=config_overwrite,
    #         )

    #     with self.subTest("Test TP only"):
    #         tp_size = 2
    #         pp_size = 1
    #         test_fn(tp_size, pp_size)

    #     is_pp_supported = ParallelizersManager.parallelizer_for_model(model_type).supports_pipeline_parallelism()
    #     if is_pp_supported:
    #         with self.subTest("Test PP only"):
    #             tp_size = 1
    #             pp_size = 2
    #             test_fn(tp_size, pp_size)

    #         with self.subTest("Test TP + PP only"):
    #             tp_size = 2
    #             pp_size = 4
    #             test_fn(tp_size, pp_size)

    # @parameterized.expand(MODELS_TO_TEST)
    # def test_model_parallel_from_pretrained_lazy_load(
    #     self, model_class_name: str, model_name_or_path: str, config_overwrite: Dict[str, str]
    # ):
    #     # In this test, we:
    #     #   1. Test parallelism when initializing lazily from pretrained weights.
    #     #   2. Enable embedding parallelization.
    #     #   3. Enable sequence parallelism.
    #     def test_fn(tp_size: int, pp_size: int):
    #         self._test_model_parallel(
    #             tp_size=tp_size,
    #             pp_size=pp_size,
    #             num_neuron_cores=8,
    #             run_test_in_parallel=True,
    #             model_class_name=model_class_name,
    #             model_name_or_path=model_name_or_path,
    #             from_config=False,
    #             with_lazy_load=True,
    #             parallelize_embeddings=True,
    #             sequence_parallel_enabled=True,
    #             overwrite_model_config=config_overwrite,
    #         )

    #     with self.subTest("Test TP only"):
    #         tp_size = 2
    #         pp_size = 1
    #         test_fn(tp_size, pp_size)

    #     is_pp_supported = ParallelizersManager.parallelizer_for_model(model_type).supports_pipeline_parallelism()
    #     if is_pp_supported:
    #         with self.subTest("Test PP only"):
    #             tp_size = 1
    #             pp_size = 2
    #             test_fn(tp_size, pp_size)

    #         with self.subTest("Test TP + PP only"):
    #             tp_size = 2
    #             pp_size = 4
    #             test_fn(tp_size, pp_size)

    # @pytest.mark.skipif(
    #     NUM_NEURON_CORES_AVAILABLE < 32,
    #     reason=f"This test requires 32 Neuron cores, but only {NUM_NEURON_CORES_AVAILABLE} are available",
    # )
    # def test_llama_v2_gqa_variants(self):
    #     llama_v2_model_name = "anushehchaudry/llama-2-tiny-random"
    #     # MHA setup
    #     # TP size = 2, num_attention_heads = 8, num_key_value_heads = 8
    #     self._test_model_parallel(
    #         tp_size=2,
    #         pp_size=1,
    #         num_neuron_cores=8,
    #         run_test_in_parallel=True,
    #         model_class_name="LlamaForCausalLM",
    #         model_name_or_path=llama_v2_model_name,
    #         from_config=True,
    #         with_lazy_load=False,
    #         parallelize_embeddings=False,
    #         sequence_parallel_enabled=False,
    #         overwrite_model_config={
    #             "num_hidden_layers": "2",
    #             "num_attention_heads": "8",
    #             "num_key_value_heads": "8",
    #         },
    #     )

    #     # GQA setup with num_key_value_heads > tp_size.
    #     # TP size = 2, num_attention_heads = 8, num_key_value_heads = 4
    #     self._test_model_parallel(
    #         tp_size=2,
    #         pp_size=1,
    #         num_neuron_cores=8,
    #         run_test_in_parallel=True,
    #         model_class_name="LlamaForCausalLM",
    #         model_name_or_path=llama_v2_model_name,
    #         from_config=True,
    #         with_lazy_load=False,
    #         parallelize_embeddings=False,
    #         sequence_parallel_enabled=False,
    #         overwrite_model_config={
    #             "num_hidden_layers": "2",
    #             "num_attention_heads": "8",
    #             "num_key_value_heads": "4",
    #         },
    #     )

    #     # GQA setup with num_key_value_heads = tp_size.
    #     # TP size = 8, num_attention_heads = 16, num_key_value_heads = 8
    #     self._test_model_parallel(
    #         tp_size=8,
    #         pp_size=1,
    #         num_neuron_cores=8,
    #         run_test_in_parallel=True,
    #         model_class_name="LlamaForCausalLM",
    #         model_name_or_path=llama_v2_model_name,
    #         from_config=True,
    #         with_lazy_load=False,
    #         parallelize_embeddings=False,
    #         sequence_parallel_enabled=False,
    #         overwrite_model_config={
    #             "num_hidden_layers": "2",
    #             "hidden_size": "32",
    #             "num_attention_heads": "16",
    #             "num_key_value_heads": "8",
    #         },
    #     )

    #     # GQA setup with num_key_value_heads < tp_size.
    #     # TP size = 8, num_attention_heads = 16, num_key_value_heads = 2
    #     self._test_model_parallel(
    #         tp_size=8,
    #         pp_size=1,
    #         num_neuron_cores=8,
    #         run_test_in_parallel=True,
    #         model_class_name="LlamaForCausalLM",
    #         model_name_or_path=llama_v2_model_name,
    #         from_config=True,
    #         with_lazy_load=False,
    #         parallelize_embeddings=False,
    #         sequence_parallel_enabled=False,
    #         overwrite_model_config={
    #             "num_hidden_layers": "2",
    #             "hidden_size": "32",
    #             "num_attention_heads": "16",
    #             "num_key_value_heads": "2",
    #         },
    #     )

    #     # MQA setup
    #     # TP size = 8, num_attention_heads = 16, num_key_value_heads = 1
    #     self._test_model_parallel(
    #         tp_size=8,
    #         pp_size=1,
    #         num_neuron_cores=8,
    #         run_test_in_parallel=True,
    #         model_class_name="LlamaForCausalLM",
    #         model_name_or_path=llama_v2_model_name,
    #         from_config=True,
    #         with_lazy_load=False,
    #         parallelize_embeddings=False,
    #         sequence_parallel_enabled=False,
    #         overwrite_model_config={
    #             "num_hidden_layers": "2",
    #             "hidden_size": "32",
    #             "num_attention_heads": "16",
    #             "num_key_value_heads": "1",
    #         },
    #     )
