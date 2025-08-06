import logging
import os

from functools import partial
from typing import List, Optional, Union, Type

import torch

from optimum.neuron.models.neuron_config import NxDNeuronConfig


from transformers import AutoConfig, AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, set_seed
from transformers.generation import SampleDecoderOnlyOutput, SampleEncoderDecoderOutput
from transformers.image_utils import ImageInput

from torch_neuronx.testing.validation import logit_validation
import neuronx_distributed as nxd



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

    Arguments:
        model_path (str) - The path to the pre-trained model.
        tokenizer_path (str) - The path to the tokenizer associated with the model.
        generation_config (GenerationConfig) - Configuration settings for text generation tasks.
    """

    def __init__(self, model_path: str, tokenizer_path: str = None, generation_config: GenerationConfig = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.hf_config = None

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

    def generate_quantized_hf_checkpoints_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        hf_config = self.get_hf_config(sequence_length=sequence_length, **kwargs)
        neuron_config = self.get_config_for_nxd(
            hf_config,
            batch_size,
            1,
            max_prompt_length,
            sequence_length,
            enable_bucketing=False,
            **kwargs)
        neuron_config.hf_config.torch_dtype = torch.float32
        quantized_state_dict = self.get_model_cls().generate_quantized_state_dict(
            model_path=self.model_path, neuron_config=neuron_config
        )
        return quantized_state_dict

    def load_neuron_model(self, traced_model_path, start_rank_id=None, local_ranks_size=None):
        neuron_config = self.get_config_cls().load(traced_model_path)
        model = self.get_model_cls().from_pretrained("", neuron_config)
        model.load(traced_model_path, start_rank_id=start_rank_id, local_ranks_size=local_ranks_size)
        if neuron_config.hf_config.torch_dtype == torch.bfloat16:
            model.bfloat16()
        return model

    def load_tokenizer(self, padding_side=None):
        # Implement per model
        raise NotImplementedError

    def load_image_processor(self):
        return None

    def load_processor(self):
        return None

    def get_config_cls(self) -> Type[NxDNeuronConfig]:
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

    def init_distributed_env(self):
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

    def get_hf_config(self, sequence_length):
        if self.hf_config is not None:
            return self.hf_config

        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            self.model_path,
            do_sample=self.generation_config.do_sample,
            top_k=self.generation_config.top_k,
            pad_token_id=self.generation_config.pad_token_id
        )
        hf_config.max_length = sequence_length
        if hasattr(hf_config, "max_position_embeddings") and hf_config.max_position_embeddings <= hf_config.max_length:
            logging.warning(
                "max_position_embeddings is less than or equal to max_length. Updating max_position_embeddings..."
            )
            hf_config.max_position_embeddings = hf_config.max_length + 1  # otherwise get error
        self.hf_config = hf_config
        return hf_config

    def get_config_for_nxd(
        self,
        hf_config,
        batch_size,
        tp_degree,
        max_prompt_length,
        sequence_length,
        enable_bucketing,
        **kwargs,
    ) -> NxDNeuronConfig:
        """
        Set up the value for config attributes if needed.

        Please don't add new config attribute here. Instead, please add new
        attributes in NeuronConfig or model-specific config class.
        """
        config_cls = self.get_config_cls()
        try:
            neuron_config = config_cls.load(self.model_path, skip_hf_config=True, **kwargs)
            neuron_config.hf_config = hf_config
            return neuron_config
        except FileNotFoundError:
            return config_cls(hf_config=hf_config,
                              tp_degree=tp_degree,
                              batch_size=batch_size,
                              seq_len=sequence_length,
                              padding_side=self.get_padding_side(),
                              max_context_length=max_prompt_length,
                              enable_bucketing=enable_bucketing,
                              **kwargs)

    def generate_with_hf(self, prompts: List[str], max_length: int, **kwargs):
        """
        Use this to generate CPU goldens against which the trace is validated.
        """
        model = self.load_hf_model()
        if kwargs.get("images") is not None:
            processor = self.load_processor(padding_side="left")
            tokenizer = processor.tokenizer
            kwargs["image_processor"] = processor.image_processor
        else:
            tokenizer = self.load_tokenizer(padding_side="left")
        return self.generate(model, tokenizer, prompts, max_length, **kwargs)

    def generate_on_neuron(self, prompts: List[str], model: PreTrainedModel, draft_model: PreTrainedModel = None, **kwargs):
        """
        Runs the trace on Neuron.
        """

        if not isinstance(model, PreTrainedModel):
            raise ValueError(f"Model should be of type PreTrainedModel, got type {type(model)}")

        if kwargs.get("images") is not None:
            processor = self.load_processor()
            tokenizer = processor.tokenizer
            kwargs["image_processor"] = processor.image_processor
        else:
            tokenizer = self.load_tokenizer()

        if len(prompts) != model.neuron_config.max_batch_size:
            raise ValueError(f"Number of prompts should match batch size {model.neuron_config.max_batch_size}")

        max_length = kwargs.pop("max_length", model.neuron_config.max_length)
        if (max_length > model.neuron_config.max_length):
            ValueError(f"Found user supplied {max_length=} exceeds the compiled model sequence_length={model.neuron_config.max_length}")

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

        if kwargs.get("images"):
            processor = self.load_processor()
            tokenizer = processor.tokenizer
            kwargs["image_processor"] = processor.image_processor
        else:
            tokenizer = self.load_tokenizer()
        outputs, output_tokens = self.generate(model, tokenizer, prompts, sequence_length, **kwargs)
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

        # If pixel_values is given, pass to the model
        # Else generate pixel_values from given images
        images = kwargs.pop("images", None)
        pixel_values = kwargs.get("pixel_values")
        if images is not None and pixel_values is None:
            image_processor = kwargs.pop("image_processor", None)
            assert image_processor is not None, "image_processor is required when passing images"
            pixel_values = image_processor(images, return_tensors="pt")["pixel_values"]
            kwargs["pixel_values"] = pixel_values

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
        expected_token_ids: Optional[List] = None,
        on_cpu: bool = False,
        do_sample: bool = True,
        traced_draft_model: PreTrainedModel = None,
        speculation_length: int = 0,
        prompt: Optional[str] = None,
        image: Optional[ImageInput] = None,
        **kwargs,
    ):
        """
        Function to compare outputs from huggingface model and neuronx NxD model
        """
        if prompt is None:
            prompts = [TEST_PROMPT] * batch_size
        else:
            prompts = [prompt] * batch_size

        if image is not None:
            kwargs["images"] = [image] * batch_size

        tokenizer = self.load_tokenizer()

        if expected_token_ids is not None:
            outputs_expected = tokenizer.batch_decode(
                expected_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        else:
            # Generate goldens with HF on CPU
            expected_token_ids, outputs_expected = self.generate_with_hf(
                prompts, max_length, do_sample=do_sample, **kwargs
            )
        print(f"Expected output: {outputs_expected}")

        # Generate outputs with NxD
        print("Generating outputs with NxD")
        if on_cpu:
            max_prompt_length = kwargs.pop("max_prompt_length")
            output_token_ids, outputs_actual = self.generate_on_cpu(
                prompts,
                batch_size,
                max_prompt_length=max_prompt_length,
                sequence_length=max_length,
                **kwargs
            )
        else:
            output_token_ids, outputs_actual = self.generate_on_neuron(
                prompts, traced_model, traced_draft_model, do_sample=do_sample, max_length=max_length,  **kwargs
            )
        print(f"Actual output  : {outputs_actual}")

        pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token) if tokenizer else 0
        output_token_ids = output_token_ids[output_token_ids != pad_token_id]
        expected_token_ids = expected_token_ids[expected_token_ids != pad_token_id]
        if traced_draft_model is not None:
            # Handle corner scenario where last few tokens are not generated as part of speculation.
            assert (
                abs(expected_token_ids.shape[-1] - output_token_ids.shape[-1]) <= speculation_length
            ), "Unexpected number of tokens generated by target model"
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
        if traced_model.neuron_config.on_device_sampling:
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
    ):
        """
        Function to trace a model with neuronx NxD
        """
        if traced_model_path is not None:
            if not os.path.exists(traced_model_path):
                os.makedirs(traced_model_path)

        # Write the model config into the traced_model_path
        hf_config = self.get_hf_config(sequence_length=sequence_length)
        if hf_config.torch_dtype != torch.float32 and hf_config.torch_dtype != torch.bfloat16:
            raise ValueError(
                f"Type {hf_config.torch_dtype} is not supported for this model at this time. Please choose float32 or bfloat16."
            )


        self.neuron_config = self.get_config_for_nxd(
            hf_config,
            batch_size,
            tp_degree,
            max_prompt_length,
            sequence_length,
            enable_bucketing,
        )

        # Write the model config into the traced_model_path
        self.neuron_config.save(traced_model_path)

        # Copy the tokenizer into the traced_model_path
        tokenizer = self.load_tokenizer()
        if tokenizer:
            tokenizer.save_pretrained(traced_model_path)

        model = self.get_model_cls().from_pretrained(self.model_path, self.neuron_config)

        model.compile(serialize_base_path=traced_model_path)
