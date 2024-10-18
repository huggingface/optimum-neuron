import logging
import os
from typing import List, Union

import torch
from transformers import AutoTokenizer, GenerationConfig, PreTrainedModel, set_seed
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput


CONTEXT_ENCODING_MODEL = "context_encoding_model"
TOKEN_GENERATION_MODEL = "token_generation_model"


BASE_COMPILER_WORK_DIR = "/tmp/nxd_model/"
CTX_ENC_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + CONTEXT_ENCODING_MODEL + "/"
TKN_GEN_MODEL_COMPILER_WORK_DIR = BASE_COMPILER_WORK_DIR + TOKEN_GENERATION_MODEL + "/"


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


class InferenceRunner:
    """
    Use the runner class to trace the model and perform inference.

    Usage:
        trace - Traces the neuron wrapper
        infer - Runs the traced model on Neuron
        infer-on-cpu - Runs the neuron wrapper on CPU
        infer-with-hf - Runs inference with huggingface model on CPU
    """

    def __init__(self, model_path: str = None, generation_config: GenerationConfig = None):
        self.model_path = model_path

        if generation_config is None:
            generation_config = GenerationConfig.from_pretrained(model_path)
            generation_config.top_k = 1
            generation_config.do_sample = True
            generation_config.pad_token_id = 0

        self.generation_config = generation_config

    def load_neuron_model(self, traced_model_path):
        # Implement per model
        raise NotImplementedError

    def get_config_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_model_cls(self):
        # Implement per model
        raise NotImplementedError

    def get_default_hf_generation_config_kwargs(self) -> dict:
        return {
            "do_sample": self.generation_config.do_sample,
            "top_k": self.generation_config.top_k,
            "pad_token_id": self.generation_config.pad_token_id,
        }

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

        if config.max_position_embeddings <= max_length:
            logging.warning(
                "max_position_embeddings is less than or equal to max_length. Updating max_position_embeddings..."
            )
            config.max_position_embeddings = max_length + 1  # otherwise get error

        config.max_batch_size = batch_size
        config.batch_size = batch_size

        # bucketing specific
        config.enable_bucketing = enable_bucketing

        config.padding_side = "right"
        config.on_device_sampling = kwargs.get("on_device_sampling", False)

        config.trace_tokengen_model = kwargs.get("trace_tokengen_model", True)

        config.pad_token_id = kwargs.get("pad_token_id", None)

        return config

    def generate(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        max_length: int = None,
        **kwargs,
    ):
        # Sanity checks
        if len(prompts) != model.config.max_batch_size:
            raise ValueError(f"Number of prompts should match batch size {model.config.max_batch_size}")
        if max_length is None:
            max_length = model.config.max_length
        if max_length > model.config.max_length:
            ValueError(
                f"Found user supplied {max_length=} exceeds the compiled model sequence_length={model.config.max_length}"
            )
        set_seed(0)  # to avoid randomness in sampling if any
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        for idx, input in enumerate(inputs["input_ids"]):
            logging.debug("tokenized input %s : %s", idx, tokenizer.decode(input))

        outputs = model.generate(
            inputs.input_ids,
            generation_config=self.generation_config,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            **kwargs,
        )
        model.reset()

        if isinstance(outputs, SampleOutput.__args__):
            # Get token ids from output when return_dict_in_generate=True
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_tokens = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return outputs, output_tokens

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
        if sequence_length <= max_prompt_length:
            raise ValueError(
                f"Found {sequence_length=} which is less than or equal to {max_prompt_length=}. Please make sure sequence_length is strictly greater than max_prompt_length"
            )

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

        model = self.get_model_cls().from_pretrained(self.model_path, config)

        model.compile(serialize_base_path=traced_model_path)
