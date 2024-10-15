import json
import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import List, Union

import torch
from modules.benchmark import BENCHMARK_REPORT_FILENAME, Benchmark, LatencyCollector, generate_report
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer, GenerationConfig, PreTrainedModel, set_seed
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput

from torch_neuronx.testing.validation import logit_validation
import neuronx_distributed as nxd

END_TO_END_MODEL = "e2e_model"
CONTEXT_ENCODING_MODEL = "context_encoding_model"
TOKEN_GENERATION_MODEL = "token_generation_model"
LM_HEAD_NAME = "lm_head.pt"


BASE_COMPILER_WORK_DIR = "/tmp/nxd_model/"
CTX_ENC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + CONTEXT_ENCODING_MODEL + "/"
TKN_GEN_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + TOKEN_GENERATION_MODEL + "/"


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]

TEST_PROMPT = "I believe the meaning of life is"


class InferenceRunner:
    """
    Use the runner class to trace the model and perform inference.

    Usage:
        trace - Traces the neuron wrapper
        infer - Runs the traced model on Neuron
        infer-on-cpu - Runs the neuron wrapper on CPU
        infer-with-hf - Runs inference with huggingface model on CPU
    """

    def __init__(self, model_path: str = None, tokenizer_path: str = None, generation_config: GenerationConfig = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self._is_torch_profile_enabled = False

        if generation_config is None:
            generation_config = GenerationConfig.from_pretrained(model_path)
            generation_config.top_k = 1
            generation_config.do_sample = True
            generation_config.pad_token_id = 0

        self.generation_config = generation_config

    def load_hf_model(self):
        # Implement per model
        raise NotImplementedError

    def load_neuron_model_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        # Implement per model
        raise NotImplementedError

    def load_neuron_model(self, traced_model_path):
        # Implement per model
        raise NotImplementedError

    def load_tokenizer(self, padding_side=None):
        # Implement per model
        raise NotImplementedError

    def get_config_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_model_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_padding_side(self):
        # Implement per model
        raise NotImplementedError

    def get_default_hf_generation_config_kwargs(self) -> dict:
        return {
            'do_sample': self.generation_config.do_sample,
            'top_k': self.generation_config.top_k,
            'pad_token_id': self.generation_config.pad_token_id
        }

    def enable_torch_profile(self):
        self._is_torch_profile_enabled = True

    def is_torch_profile_enabled(self):
        return self._is_torch_profile_enabled

    @contextmanager
    def torch_profile(self, chrome_trace_path: str = "torch-trace.json", **profile_kwargs):
        if self.is_torch_profile_enabled():
            with profile(activities=[ProfilerActivity.CPU], **profile_kwargs) as prof:
                yield prof
            prof.export_chrome_trace(chrome_trace_path)
        else:
            yield

    def init_ditributed_env(self):
        """
        Initialize a simple neuronx distributed (Tensor Parallelism) environment, where there TP degree is 1.

        This function is just for running NeuronxDistributed models on CPU to validate correctness.
        """
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2024"

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="xla")

        nxd.parallel_layers.parallel_state.destroy_model_parallel()
        nxd.parallel_layers.parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)

    def get_config_for_nxd(
        self,
        batch_size,
        tp_degree,
        max_prompt_length,
        sequence_length,
        enable_bucketing,
        **kwargs,
    ):
        """
        Set up the value for config attributes if needed.

        Please don't add new config attribute here. Instead, please add new
        attributes in NeuronInferenceConfig or model-specific config class.
        """
        config_cls = self.get_config_cls()

        merged_kwargs = self.get_default_hf_generation_config_kwargs()
        if kwargs is not None:
            merged_kwargs.update(kwargs)
        config = config_cls.from_pretrained(self.model_path, **merged_kwargs)

        config.tp_degree = tp_degree

        config.max_context_length = max_prompt_length
        config.max_new_tokens = sequence_length - max_prompt_length
        if config.max_new_tokens == 0:
            config.max_new_tokens = None
        max_length = sequence_length
        config.max_length = max_length
        config.n_positions = max_length
        config.n_active_tokens = max_length

        if config.max_position_embeddings <= max_length:
            logging.warning(
                "max_position_embeddings is less than or equal to max_length. Updating max_position_embeddings..."
            )
            config.max_position_embeddings = max_length + 1  # otherwise get error

        config.max_batch_size = batch_size
        config.ctx_batch_size = batch_size
        config.tkg_batch_size = batch_size
        config.batch_size = batch_size

        # bucketing specific
        config.enable_bucketing = enable_bucketing
        config.buckets = [max_length]

        config.padding_side = self.get_padding_side()
        config.on_device_sampling = kwargs.get("on_device_sampling", False)

        config.trace_tokengen_model = kwargs.get("trace_tokengen_model", True)

        config.pad_token_id = kwargs.get("pad_token_id", None)

        return config

    def generate_with_hf(self, prompts: List[str], max_length: int, **kwargs):
        """
        Use this to generate CPU goldens against which the trace is validated.
        """
        model = self.load_hf_model()
        tokenizer = self.load_tokenizer(padding_side="left")
        return self.generate(model, tokenizer, prompts, max_length, **kwargs)

    def generate_on_neuron(self, prompts: List[str], model: PreTrainedModel, draft_model: PreTrainedModel = None, **kwargs):
        """
        Runs the trace on Neuron.
        """

        if not isinstance(model, PreTrainedModel):
            raise ValueError(f"Model should be of type PreTrainedModel, got type {type(model)}")

        tokenizer = self.load_tokenizer()
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if len(prompts) != model.config.max_batch_size:
            raise ValueError(f"Number of prompts should match batch size {model.config.max_batch_size}")

        max_length = kwargs.pop("max_length", model.config.max_length)
        if (max_length > model.config.max_length):
            ValueError(f"Found user supplied {max_length=} exceeds the compiled model sequence_length={model.config.max_length}")

        with self.torch_profile(chrome_trace_path="generate-on-neuron.torch-trace.json"):
            outputs, output_tokens = self.generate(
                model, tokenizer, prompts, max_length, draft_model, **kwargs
            )
        model.reset()
        if draft_model is not None:
            draft_model.reset()
        return outputs, output_tokens

    def generate_on_cpu(self, prompts: List[str], batch_size: int, max_prompt_length: int, sequence_length: int, **kwargs):
        """
        Use generate_on_cpu to confirm the neuron wrapper is correct. If the wrapper works
        on CPU, then the trace should work too. If it does not, it indicates a problem with
        the trace itself.
        """
        model = self.load_neuron_model_on_cpu(max_prompt_length, sequence_length, batch_size, **kwargs)

        tokenizer = self.load_tokenizer()
        outputs, output_tokens = self.generate(model, tokenizer, prompts, sequence_length)
        model.reset()
        return outputs, output_tokens

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        max_length: int,
        draft_model: PreTrainedModel = None,
        **kwargs
    ):
        set_seed(0)  # to avoid randomness in sampling if any
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        for idx, input in enumerate(inputs["input_ids"]):
            logging.debug("tokenized input %s : %s", idx, tokenizer.decode(input))

        if draft_model is not None:
            kwargs.update({
                "assistant_model": draft_model,
                "do_sample": False
            })

        outputs = model.generate(
            inputs.input_ids,
            generation_config=self.generation_config,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            **kwargs,
        )

        if isinstance(outputs, SampleOutput.__args__):
            # Get token ids from output when return_dict_in_generate=True
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_tokens = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs, output_tokens

    def check_accuracy(
        self,
        traced_model: PreTrainedModel,
        batch_size: int,
        max_length: int,
        expected_token_ids: List = None,
        on_cpu: bool = False,
        do_sample: bool = True,
        traced_draft_model: PreTrainedModel = None,
        **kwargs,
    ):
        """
        Function to compare outputs from huggingface model and neuronx NxD model
        """
        prompts = [TEST_PROMPT] * batch_size
        tokenizer = self.load_tokenizer()

        if expected_token_ids is not None:
            outputs_expected = tokenizer.batch_decode(
                expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            # Generate goldens with HF on CPU
            expected_token_ids, outputs_expected = self.generate_with_hf(
                prompts, max_length, do_sample=do_sample
            )
        print(f"Expected output: {outputs_expected}")

        # Generate outputs with NxD
        if on_cpu:
            max_prompt_length = kwargs.pop("max_prompt_length")
            output_token_ids, outputs_actual = self.generate_on_cpu(
                prompts,
                batch_size,
                max_prompt_length=max_prompt_length,
                sequence_length=max_length
            )
        else:
            output_token_ids, outputs_actual = self.generate_on_neuron(
                prompts, traced_model, traced_draft_model, do_sample=do_sample, max_length=max_length
            )
        print(f"Actual output  : {outputs_actual}")

        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if tokenizer else 0
        output_token_ids = output_token_ids[output_token_ids != pad_token_id]
        expected_token_ids = expected_token_ids[expected_token_ids != pad_token_id]
        if traced_draft_model is not None:
            # Handle corner scenario where last few tokens are not generated as part of speculation.
            tokens_to_compare = min(expected_token_ids.shape[-1], output_token_ids.shape[-1])
            expected_token_ids = expected_token_ids[: tokens_to_compare]
            output_token_ids = output_token_ids[: tokens_to_compare]

        device = "cpu" if on_cpu else "neuron"
        assert torch.equal(
            output_token_ids, expected_token_ids
        ), f"\nActual: ({device}) {output_token_ids} \nExpected (hf-cpu): {expected_token_ids}"
        print(f"The output from Neuronx NxD on {device} is accurate!")

    def check_accuracy_logits(
        self,
        traced_model: PreTrainedModel,
        batch_size: int,
        max_length: int,
        expected_logits: torch.Tensor = None,
        divergence_difference_tol: float = 0.001,
        remove_shift: bool = True,
        tol_map: dict = None,
    ):
        if traced_model.config.on_device_sampling:
            raise ValueError("Logits validation is not supported with on-device sampling.")

        prompts = [TEST_PROMPT] * batch_size
        tokenizer = self.load_tokenizer()
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")

        if not expected_logits:
            # logit_validation assumes greedy sampling
            expected_outputs, _ = self.generate_with_hf(
                prompts, max_length, do_sample=False, output_logits=True, return_dict_in_generate=True,
            )
            expected_logits = torch.stack(expected_outputs.logits)
        expected_token_ids = expected_logits.argmax(dim=2).T
        expected_tokens = tokenizer.batch_decode(
            expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("Expected Output: ", expected_tokens, expected_token_ids)
        print("Expected Logits Shape: ", expected_logits.shape)

        def generate_logits(model, tokenizer, input_ids):
            prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            actual_outputs, actual_tokens = self.generate_on_neuron(
                prompt, traced_model, do_sample=True, output_logits=True, return_dict_in_generate=True,
                max_length=max_length
            )
            actual_logits = torch.stack(actual_outputs.logits)
            actual_token_ids = actual_logits.argmax(dim=2).T
            print("Actual Output: ", actual_tokens, actual_token_ids)
            print("Actual Logits Shape: ", actual_logits.shape)
            model.reset()
            return actual_logits

        generate_fn = partial(generate_logits, traced_model, tokenizer)
        passed, result, status_msg = logit_validation(inputs.input_ids,
                                                      generate_fn,
                                                      expected_logits,
                                                      divergence_difference_tol=divergence_difference_tol,
                                                      tol_map=tol_map,
                                                      pad_token_id=tokenizer.pad_token_id,
                                                      padding_side=tokenizer.padding_side)
        assert passed, status_msg
        print("Passed logits validation")

    def trace(
        self,
        traced_model_path,
        tp_degree,
        batch_size,
        max_prompt_length=128,
        sequence_length=256,
        enable_bucketing=True,
        **kwargs,
    ):
        """
        Function to trace a model with neuronx NxD
        """
        if (sequence_length <= max_prompt_length):
            raise ValueError(f"Found {sequence_length=} which is less than or equal to {max_prompt_length=}. Please make sure sequence_length is strictly greater than max_prompt_length")

        if traced_model_path is not None:
            if not os.path.exists(traced_model_path):
                os.makedirs(traced_model_path)

        # Write the model config into the traced_model_path
        config = self.get_config_for_nxd(
            batch_size,
            tp_degree,
            max_prompt_length,
            sequence_length,
            enable_bucketing,
            **kwargs,
        )
        if config.torch_dtype != torch.float32 and config.torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Type {config.torch_dtype} is not supported for this model at this time. Please choose float32 or bfloat16."
            )
        # We have the config in the trace_model_path
        config.save_pretrained(traced_model_path)

        # Save config to be used by checkpoint_loader
        self.config = config

        # Copy the tokenizer into the traced_model_path
        tokenizer = self.load_tokenizer()
        if tokenizer:
            tokenizer.save_pretrained(traced_model_path)

        model = self.get_model_cls().from_pretrained(self.model_path, config)

        model.compile(serialize_base_path=traced_model_path)

    def benchmark_sampling(self, model: PreTrainedModel, draft_model: PreTrainedModel = None, target: str = None):
        config = model.config
        tokenizer = self.load_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token

        target = target if target is not None else "all"

        report = {}

        # Benchmark E2E model
        if target in ["all", "e2e"]:
            batch_encoding = self.get_sample_inputs(END_TO_END_MODEL, config, tokenizer)
            input_param = {
                "input_ids": batch_encoding["input_ids"],
                "attention_mask": batch_encoding["attention_mask"],
                "max_new_tokens": config.max_new_tokens,
                "top_k": 1,
                "do_sample": draft_model is None,
                "assistant_model": draft_model,
            }

            if target == "all":
                latency_collectors = self.create_submodule_latency_collectors(model)

            # Register latency collectors after warm-up to avoid recording warm-up metrics.
            def register_latency_collectors():
                if target == "all":
                    self.register_latency_collectors(latency_collectors, model)

            e2e_benchmark = Benchmark(model.generate, input_param, config, preprocess_func=model.reset,
                                      post_warmup_func=register_latency_collectors)
            e2e_benchmark.run()
            report[END_TO_END_MODEL] = generate_report(e2e_benchmark.latency_list, config)

            if target == "all":
                report.update(self.generate_submodule_reports(latency_collectors, config))

        # Benchmark context encoding model only
        if target == "context_encode":
            input_param = self.get_sample_inputs(CONTEXT_ENCODING_MODEL, config)
            ctx_enc_benchmark = Benchmark(model.context_encoding_model, input_param, config)
            ctx_enc_benchmark.run()
            report[CONTEXT_ENCODING_MODEL] = generate_report(ctx_enc_benchmark.latency_list, config)

        # Benchmark token generation model only
        if hasattr(model, "token_generation_model") and target == "token_gen":
            input_param = self.get_sample_inputs(TOKEN_GENERATION_MODEL, config)
            tkn_gen_benchmark = Benchmark(model.token_generation_model, input_param, config)
            tkn_gen_benchmark.run()
            report[TOKEN_GENERATION_MODEL] = generate_report(tkn_gen_benchmark.latency_list, config)

        model.reset()
        if draft_model is not None:
            draft_model.reset()

        print("Benchmark completed and its result is as following")
        print(json.dumps(report, indent=4))
        with open(BENCHMARK_REPORT_FILENAME, "w") as f:
            json.dump(report, f)
        print("Completed saving result to " + BENCHMARK_REPORT_FILENAME)

        return report

    def get_sample_inputs(self, model_type, config, tokenizer=None):
        max_length = config.max_length
        batch_size = config.batch_size

        sample_inputs = None
        if model_type == END_TO_END_MODEL:
            sample_inputs = tokenizer(
                [TEST_PROMPT] * batch_size,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        elif model_type == CONTEXT_ENCODING_MODEL:
            input_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, max_length), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)

            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )
        elif model_type == TOKEN_GENERATION_MODEL:
            input_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, max_length), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, 1), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)
            sample_inputs = (
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )

        return sample_inputs

    def create_submodule_latency_collectors(self, model):
        collectors = {}
        collectors[CONTEXT_ENCODING_MODEL] = LatencyCollector()
        if hasattr(model, "token_generation_model"):
            collectors[TOKEN_GENERATION_MODEL] = LatencyCollector()
        return collectors

    def register_latency_collectors(self, latency_collectors, model):
        self.register_forward_latency_collector(latency_collectors[CONTEXT_ENCODING_MODEL],
                                                model.context_encoding_model)
        if TOKEN_GENERATION_MODEL in latency_collectors:
            self.register_forward_latency_collector(latency_collectors[TOKEN_GENERATION_MODEL],
                                                    model.token_generation_model)

    def register_forward_latency_collector(self, latency_collector, model):
        model.register_forward_pre_hook(latency_collector.pre_hook)
        model.register_forward_hook(latency_collector.hook)

    def generate_submodule_reports(self, latency_collectors, config):
        return {key : generate_report(collector.latency_list, config) for key, collector in latency_collectors.items()}
