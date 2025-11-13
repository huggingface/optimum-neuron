# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import inspect
from collections import defaultdict, deque
from typing import Any, Callable

import datasets
import numpy as np
import torch
from accelerate.utils import set_seed
from optimum.utils import logging
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)

from ..models.training import NeuronModelForCausalLM
from ..peft import NeuronPeftModel, get_peft_model
from ..utils import is_precompilation, is_trl_available
from ..utils.import_utils import is_peft_available
from .grpo_config import NeuronGRPOConfig
from .training_args import NeuronTrainingArguments
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import is_conversational
    from trl.trainer.utils import disable_dropout_in_model, identity, nanmax, nanmin, nanstd
else:

    class GRPOTrainer:
        pass

    class GRPOConfig:
        pass


if is_peft_available():
    from peft import PeftConfig
else:

    class PeftConfig:
        pass


# Create a new class that inherits from NeuronTrainer to use this class instead of the transformers Trainer,
# but has the same methods and attributes as GRPOTrainer.
# We can then inherit from this class to create our NeuronGRPOTrainer.
_GRPOTrainer = type(
    "_GRPOTrainer",
    (NeuronTrainer,),
    GRPOTrainer.__dict__.copy(),
)


logger = logging.get_logger()


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = str | PreTrainedModel | Callable[[list, list], list[float]]


def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    max_length: int | None = None,
  ) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.
    It differs from `trl` by enfoncing the same sequence length for all tensors, which is required to avoid 
    recompilation.
    """
    batch_size = len(tensors)
    if max_length is None:
        max_length = np.max([t.shape[0] for t in tensors]).tolist()

    output_shape = (max_length,) + tensors[0].shape[1:]

    # Create an output tensor filled with the padding value
    output = torch.full((batch_size, *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def neuron_parallel_compile_tokenizer_decoder_method(
    self,
    token_ids: int | list[int],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool | None = None,
    **kwargs,
) -> str:
    """
    Patched `tokenizer._decode` method for Neuron parallel compilation.
    This is needed because any tensor operation during `neuron_parallel_compile` produces rubbish results, which is not
    an issue in general, but causes failure when the token IDS end up being out of range for the tokenizer vocabulary.
    """
    if not is_precompilation():
        raise RuntimeError("This patch method should only be used with `neuron_parallel_compile`.")

    # We log the token IDs to force the data mouvement to CPU, which would happen during actual decoding.
    logger.debug("Using patched tokenizer.decode method for Neuron parallel compilation, token_ids = ", token_ids)

    # Returns a dummy string, we do not care about the value in this context.
    return "dummy"



class NeuronGRPOTrainer(_GRPOTrainer):
    """
    `GRPOTrainer` adapted for Neuron (Trainium) devices.

    This algorithm was initially proposed in the paper [DeepSeekMath: Pushing the Limits
    of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).
    """

    def __init__(
        self,
        model: str | PreTrainedModel | torch.nn.Module,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: "Dataset | IterableDataset | datasets.Dataset | None" = None,
        eval_dataset: "Dataset | dict[str, Dataset] | datasets.Dataset | None" = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        peft_config: PeftConfig | None = None,
    ):
        if not is_trl_available(required_version=TRL_VERSION):
            raise RuntimeError(f"Using NeuronGRPOTrainer requires trl=={TRL_VERSION}.")

        # Patch tokenizer decode method for Neuron parallel compilation to avoid failures.
        if is_precompilation() and hasattr(processing_class, "_decode"):
            processing_class._decode = neuron_parallel_compile_tokenizer_decoder_method.__get__(
                processing_class, processing_class.__class__
            )

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = NeuronGRPOConfig(f"{model_name}-GRPO")
        elif isinstance(args, NeuronTrainingArguments) and not isinstance(args, NeuronGRPOConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token
            dict_args.pop("push_to_hub_token", None)
            args = NeuronGRPOConfig(**dict_args)

        # Model
        if isinstance(model, str):
            model = NeuronModelForCausalLM.from_pretrained(model, args.trn_config, **args.model_init_kwargs or {})
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        model_id = model.config._name_or_path

        # Processing class
        if processing_class is None:
            from transformers import AutoProcessor

            processing_class = AutoProcessor.from_pretrained(model_id, truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Store tokens and token IDs for generation and reward computation
        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Store model forward signature keys for checking supported kwargs
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # PEFT configuration and model wrapping
        # In Prompt Tuning a small set of trainable virtual tokens (continuous prompt embeddings) is prepended to the
        # input. We store the number of these tokens so we can account for them correctly when calculating accuracy.
        self.num_virtual_tokens = 0

        if peft_config is not None and not isinstance(model, NeuronPeftModel):
            # Enable gradient checkpointing if needed
            gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
            gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs
            if args.gradient_checkpointing and (
                "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
            ):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)

            if model.active_adapter in model.peft_config:
                peft_model_config = model.peft_config[model.active_adapter]
                self.num_virtual_tokens = getattr(peft_model_config, "num_virtual_tokens", 0)

        # Reward functions - for now, only support callable reward functions
        # TODO: Add support for reward models when they can be properly loaded on Neuron
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                raise NotImplementedError(
                    "Loading reward models from model IDs is not yet implemented for NeuronGRPOTrainer. "
                    "Please provide either a PreTrainedModel or a custom callable reward function."
                )
            if isinstance(reward_func, PreTrainedModel):
                raise NotImplementedError(
                    "Using PreTrainedModel reward functions is not yet fully implemented for NeuronGRPOTrainer. "
                    "Please use a custom callable reward function for now."
                )
            # Custom callable reward function
            self.reward_func_names.append(reward_func.__name__)

        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        # Note: We skip the loop that sets up tokenizers for PreTrainedModel reward functions
        # since we currently raise errors for those anyway
        self.reward_processing_classes = reward_processing_classes

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile

        # Validate liger kernel configuration
        if self.use_liger_loss:
            raise RuntimeError(
                "Liger Kernel loss is not supported on Neuron devices. "
                "Please set use_liger_loss=False in your GRPOConfig."
            )

        # Neuron GRPO only supports vLLM generation
        if not self.use_vllm:
            raise NotImplementedError(
                "NeuronGRPOTrainer currently only supports vLLM generation. "
                "Please set use_vllm=True in your GRPOConfig."
            )

        # Only server mode is supported for now
        if self.vllm_mode != "server":
            raise NotImplementedError(
                "NeuronGRPOTrainer currently only supports vLLM server mode. "
                "Please set vllm_mode='server' in your GRPOConfig."
            )

        if self._is_vlm:
            raise NotImplementedError(
                "Vision-language models are not yet supported in NeuronGRPOTrainer. "
                "Please use text-only models for now."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            raise NotImplementedError(
                "Iterable datasets are not yet supported in NeuronGRPOTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        self._step = 0
        self._buffered_inputs = None

        # Suppress FLOP estimation warning
        model.warnings_issued["estimate_tokens"] = True

        # Initialize NeuronTrainer
        NeuronTrainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

        # Set _train_batch_size for compatibility with GRPOTrainer's get_train_dataloader
        # NeuronTrainer doesn't set this, but GRPOTrainer expects it
        self._train_batch_size = args.train_batch_size

        # Set FSDP flag to False (NeuronTrainer doesn't support FSDP)
        # GRPOTrainer's methods check this attribute
        self.is_fsdp_enabled = False

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif isinstance(model, NeuronPeftModel):
            self.ref_model = None
        else:
            # Create reference model using NeuronModelForCausalLM
            self.ref_model = NeuronModelForCausalLM.from_pretrained(model_id, args.trn_config, **args.model_init_kwargs or {})

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed
        set_seed(args.seed, device_specific=True)

        # vLLM setup - server mode only
        from ..utils import is_vllm_available

        # MOCK FLAG: Change this to False when real vLLM server is ready
        USE_MOCK_VLLM = True

        if USE_MOCK_VLLM:
            logger.warning(
                "Using MOCK vLLM client for development. This generates placeholder completions "
                "and should only be used for testing and development. Set USE_MOCK_VLLM=False in "
                "grpo_trainer.py to use real vLLM server."
            )
            from .grpo_mocks import create_mock_vllm_client

            # MOCK: Each process needs its own client (generates locally, no server)
            self.vllm_client = create_mock_vllm_client(tokenizer, args)
        else:
            if not is_vllm_available():
                raise ImportError("vLLM is not available. Please install vLLM to use NeuronGRPOTrainer.")

            # Setup vLLM server client (only on main process)
            if self.accelerator.is_main_process:
                from trl.extras.vllm_client import VLLMClient

                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"

                self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                self.vllm_client.init_communicator(device=torch.cuda.current_device())

        # vLLM specific sampling arguments
        self.guided_decoding_regex = args.vllm_guided_decoding_regex

        self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

        # Synchronize all processes after vLLM client setup
        self.accelerator.wait_for_everyone()

        # Gradient accumulation requires scaled loss
        self.model_accepts_loss_kwargs = False

        # Add tags for models
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Prepare reference model
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        # Sync reference model callback
        if args.sync_ref_model:
            from trl.trainer.callbacks import SyncRefModelCallback

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        # Prepare reward functions
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True, device_placement=True
                )

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint: Path to a checkpoint to resume from, or True to resume from the latest checkpoint.
        """
        return NeuronTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint)

    def log(self, logs: dict[str, float]) -> None:
        """
        Override GRPOTrainer's log method to use NeuronTrainer's implementation.

        GRPOTrainer has custom metrics tracking that we don't use for Neuron training.
        """
        return NeuronTrainer.log(self, logs)

    def _save_checkpoint(self, model=None, trial=None, metrics=None):
        """
        Override GRPOTrainer's _save_checkpoint to use NeuronTrainer's implementation.
        """
        return NeuronTrainer._save_checkpoint(self)

    def _prepare_inputs(self, inputs: Any) -> dict[str, Any]:
        """
        Prepare inputs for GRPO training.

        This method overrides NeuronTrainer._prepare_inputs to use GRPOTrainer's
        implementation, which handles:
        1. Generation of completions using vLLM
        2. Scoring completions using reward functions
        3. Buffering completions for reuse across multiple gradient steps
        4. Tokenization and conversion to model inputs

        Args:
            inputs: Raw batch from dataloader (list of prompt dicts for GRPO)

        Returns:
            Dictionary of tokenized tensors ready for the model
        """
        # Explicitly call GRPOTrainer's _prepare_inputs
        return GRPOTrainer._prepare_inputs(self, inputs)

    def _generate_single_turn(self, prompts: list[str], images: list | None):
        """
        Generate a single turn of completions using vLLM (mock or real server).

        This overrides GRPOTrainer's implementation to work with Neuron/XLA devices.
        The main difference is avoiding gather_object which doesn't work on XLA.

        MOCK MODE: Each process generates locally without gathering/broadcasting.
        REAL SERVER MODE: May need gather_object workaround - test when implementing!

        Args:
            prompts: List of prompt strings
            images: Optional list of images

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, forward_kwargs)
        """
        # Move model weights to vLLM if needed (no-op for mock)
        if self.state.global_step != getattr(self, "_last_loaded_step", -1):
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # For mock vLLM, generate locally on each process (no gather/broadcast needed)
        # Take unique prompts since we have num_generations duplicates
        prompts_text = [prompt if isinstance(prompt, str) else prompt["content"] for prompt in prompts]
        ordered_set_of_prompts = prompts_text[:: self.num_generations]

        if images is not None:
            ordered_set_of_images = images[:: self.num_generations]
        else:
            ordered_set_of_images = None

        # Generate using mock vLLM client
        output = self.vllm_client.generate(
            prompts=ordered_set_of_prompts,
            images=ordered_set_of_images,
            n=self.num_generations,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            min_p=0.0 if self.min_p is None else self.min_p,
            max_tokens=self.max_completion_length,
            truncate_prompt_tokens=self.max_prompt_length,
            guided_decoding_regex=self.guided_decoding_regex,
            generation_kwargs=self.args.generation_kwargs,
        )

        # Repeat prompt_ids num_generations times to match completion_ids
        prompt_ids = [ids for ids in output["prompt_ids"] for _ in range(self.num_generations)]
        completion_ids = output["completion_ids"]
        logprobs = output["logprobs"]

        # No forward_kwargs for mock vLLM
        forward_kwargs = {}

        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _generate_and_score_completions(
          self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        # We patch the pad function to make it compatible with `neuron_parallel_compile`.
        # patcher = Patcher([("trl.trainer.grpo_trainer.pad", pad)])
        # with patcher:
        return GRPOTrainer._generate_and_score_completions(self, inputs)

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ):
        """
        Override to pad sequences to max_length for XLA compilation.

        GRPO generates variable-length prompts + completions. XLA compilation requires
        fixed shapes, so we pad all sequences to max_length (max_prompt_length + max_completion_length).
        """
        # Calculate max_length from GRPO config
        max_length = self.max_prompt_length + self.max_completion_length
        seq_len = input_ids.shape[1]

        if seq_len < max_length:
            pad_amount = max_length - seq_len

            # Pad input_ids with pad_token_id
            input_ids = torch.nn.functional.pad(
                input_ids,
                (0, pad_amount),
                value=self.pad_token_id
            )

            # Pad attention_mask
            if attention_mask is not None:
                attention_mask = torch.nn.functional.pad(
                    attention_mask,
                    (0, pad_amount),
                    value=0  # Padded positions should be masked out
                )

        # Call parent implementation with padded tensors
        return GRPOTrainer._get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            num_images=num_images,
            pixel_attention_mask=pixel_attention_mask,
            image_sizes=image_sizes,
            token_type_ids=token_type_ids,
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str,torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        (
            prompt_ids_list,
            completion_ids_list,
            num_items_in_batch,
            sampling_per_token_logps_list,
            forward_kwargs,
        ) = self._generate(prompts, images)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        # TODO: handle this later.
        # self._logs["prompt"].extend(gather_object(prompts_text))
        # self._logs["completion"].extend(gather_object(completions_text))
        # for i, name in enumerate(self.reward_func_names):
        #     self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        # self._logs["advantages"].extend(all_process_advantages.tolist())

        # if images is not None:
        #     self._logs["images"].extend(gather_object(images))

        # if self.use_vllm and self.vllm_importance_sampling_correction:
        #     delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
        #     delta = delta[completion_mask.bool()]
        #     mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        #     max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
        #     self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
        #         self.accelerator.gather(mean_delta).mean().item()
        #     )
        #     self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
        #         self.accelerator.gather(max_delta).max().item()
        #     )

        #     flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
        #     min_importance_sampling_ratio = (
        #         torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        #     )
        #     mean_importance_sampling_ratio = (
        #         torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        #     )
        #     max_importance_sampling_ratio = (
        #         torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
        #     )
        #     self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
        #         nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
        #     )
        #     self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
        #         self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
        #     )
        #     self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
        #         nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
        #     )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        else:
            return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_pro
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        """
        Evaluation and prediction are not supported in NeuronGRPOTrainer.

        The trainer is designed for training only. NeuronTrainer does not provide
        evaluation loop functionality, and GRPO-specific evaluation would require
        significant additional implementation.
        """
        raise NotImplementedError(
            "Evaluation and prediction are not supported in NeuronGRPOTrainer. "
            "The trainer is designed for training only."
        )
