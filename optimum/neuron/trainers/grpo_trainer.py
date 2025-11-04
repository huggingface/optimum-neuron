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
import torch
from accelerate.utils import set_seed
from optimum.utils import logging
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)

from ..models.training import NeuronModelForCausalLM
from ..peft import NeuronPeftModel, get_peft_model
from ..utils import is_trl_available
from ..utils.import_utils import is_peft_available
from .grpo_config import NeuronGRPOConfig
from .training_args import NeuronTrainingArguments
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import GRPOConfig, GRPOTrainer
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
            model = NeuronModelForCausalLM.from_pretrained(model, **args.model_init_kwargs or {})
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
            or (isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values()))
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
        from trl.trainer.utils import identity

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

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            self.ref_model = None
        elif isinstance(model, NeuronPeftModel):
            self.ref_model = None
        else:
            # Create reference model using NeuronModelForCausalLM
            self.ref_model = NeuronModelForCausalLM.from_pretrained(model_id, **args.model_init_kwargs or {})

        # Disable dropout in the models
        if args.disable_dropout:
            from trl.trainer.utils import disable_dropout_in_model

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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for Neuron-optimized training.

        TODO: Implement GRPO-specific loss computation adapted for Neuron devices.
        """
        raise NotImplementedError(
            "compute_loss is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing GRPO-specific loss computation for Neuron devices."
        )

    def training_step(
        self, model: torch.nn.Module, inputs: dict[str, Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for Neuron-optimized training.

        TODO: Implement GRPO-specific training step adapted for Neuron devices.
        """
        raise NotImplementedError(
            "training_step is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing GRPO-specific training logic for Neuron devices."
        )

    def _prepare_inputs(self, inputs):
        """
        Prepare inputs for GRPO training, including generation and reward computation.

        TODO: Implement input preparation with Neuron-compatible generation and reward scoring.
        """
        raise NotImplementedError(
            "_prepare_inputs is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing prompt generation and reward computation for Neuron devices."
        )

    def _generate(self, prompts: list[str], images: list | None):
        """
        Generate completions for the given prompts.

        TODO: Implement Neuron-compatible text generation.
        """
        raise NotImplementedError(
            "_generate is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing generation logic compatible with Neuron devices."
        )

    def _generate_single_turn(self, prompts: list[str], images: list | None):
        """
        Generate a single turn of completions.

        TODO: Implement single-turn generation for Neuron devices.
        """
        raise NotImplementedError(
            "_generate_single_turn is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing single-turn generation for Neuron devices."
        )

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """
        Calculate rewards for the generated completions.

        TODO: Implement reward calculation compatible with Neuron devices.
        """
        raise NotImplementedError(
            "_calculate_rewards is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing reward computation for Neuron devices."
        )

    def _compute_loss(self, model, inputs):
        """
        Internal loss computation for GRPO.

        TODO: Implement GRPO loss computation for Neuron devices.
        """
        raise NotImplementedError(
            "_compute_loss is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing the core GRPO loss computation for Neuron devices."
        )

    def get_train_dataloader(self):
        """
        Get the training dataloader with GRPO-specific batching strategy.

        TODO: Implement GRPO-specific dataloader with proper batching for Neuron devices.
        """
        raise NotImplementedError(
            "get_train_dataloader is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing GRPO's custom batching strategy for Neuron devices."
        )

    def _get_train_sampler(self, dataset: Dataset | None = None):
        """
        Get the training sampler with GRPO-specific sampling strategy.

        TODO: Implement RepeatSampler strategy for GRPO on Neuron devices.
        """
        raise NotImplementedError(
            "_get_train_sampler is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing GRPO's RepeatSampler strategy for Neuron devices."
        )

    def _get_eval_sampler(self, eval_dataset):
        """
        Get the evaluation sampler.

        TODO: Implement evaluation sampler for GRPO on Neuron devices.
        """
        raise NotImplementedError(
            "_get_eval_sampler is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing the evaluation sampler for Neuron devices."
        )

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        """
        Compute per-token log probabilities and entropies.

        TODO: Implement log probability and entropy computation for Neuron devices.
        """
        raise NotImplementedError(
            "_get_per_token_logps_and_entropies is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing log probability computation for Neuron devices."
        )

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Get mask for high-entropy tokens.

        TODO: Implement entropy-based masking for Neuron devices.
        """
        raise NotImplementedError(
            "get_high_entropy_mask is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing entropy-based masking for Neuron devices."
        )

    def _set_signature_columns_if_needed(self):
        """
        Set signature columns for GRPO-specific data preprocessing.

        TODO: Implement signature column handling for Neuron devices.
        """
        raise NotImplementedError(
            "_set_signature_columns_if_needed is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing signature column handling for GRPO on Neuron devices."
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        """
        Perform a prediction step during evaluation.

        TODO: Implement prediction step for GRPO evaluation on Neuron devices.
        """
        raise NotImplementedError(
            "prediction_step is not yet implemented for NeuronGRPOTrainer. "
            "This requires implementing the prediction step for Neuron devices."
        )
