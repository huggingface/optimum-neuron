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
import torch_xla
import torch_xla.core.xla_model as xm
from accelerate.utils import set_seed
from neuronx_distributed import parallel_layers
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
)
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from optimum.utils import logging
from torch.utils.data import Dataset, IterableDataset, Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_rich_available

from ..accelerate.utils import (
    broadcast_object_to_pipeline_model_parallel_group,
    broadcast_object_to_tensor_model_parallel_group,
    gather_object_from_data_parallel_group,
)
from ..models.training import NeuronModelForCausalLM
from ..peft import NeuronPeftModel, get_peft_model
from ..peft.utils.vllm import get_original_merged_weights_for_vllm
from ..utils import is_precompilation, is_trl_available
from ..utils.import_utils import is_peft_available
from .extras import MockVLLMClient, VLLMClient
from .grpo_config import NeuronGRPOConfig
from .training_args import NeuronTrainingArguments
from .transformers import NeuronTrainer
from .trl_utils import (
    TRL_VERSION,
    DistributedRepeatSampler,
    batch_pad_sequences,
    nanmax,
    nanmin,
    nanstd,
    neuron_parallel_compile_tokenizer_decoder_method,
)


if is_wandb_available():
    import wandb

if is_trl_available():
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import is_conversational, maybe_apply_chat_template
    from trl.extras.vllm_client import VLLMClient as TRLVLLMClient
    from trl.trainer.utils import (
        RepeatSampler,
        disable_dropout_in_model,
        entropy_from_logits,
        identity,
        print_prompt_completions_sample,
        selective_log_softmax,
    )
else:

    class GRPOTrainer:
        pass

    class GRPOConfig:
        pass

    class TRLVLLMClient:
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
        vllm_client: TRLVLLMClient | None = None,
        fixed_size_for_obj_collectives: int | None = 10 * 1024 * 1024,  # 10 MB
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
            gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
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
        self.reward_weights = self.reward_weights.to(xm.xla_device())

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
            self.ref_model = NeuronModelForCausalLM.from_pretrained(
                model_id, args.trn_config, **args.model_init_kwargs or {}
            )

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
        if vllm_client is not None:
            # Use injected client (for testing, mocking, or custom implementations)
            self.vllm_client = vllm_client
        else:
            # Default: Create VLLMClient from args
            from ..utils import is_vllm_available

            if not is_vllm_available():
                raise ImportError("vLLM is not available. Please install vLLM to use NeuronGRPOTrainer.")

            if args.vllm_server_base_url is not None:
                base_url = args.vllm_server_base_url
            else:
                base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"

            # For `neuron_parallel_compile`, use a mock VLLM client that doesn't make actual server requests.
            if is_precompilation():
                self.vllm_client = MockVLLMClient(tokenizer, max_completion_length=self.max_completion_length)
            else:
                self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
            # Only main process initializes the communicator for weight updates
            if self.accelerator.is_main_process:
                self.vllm_client.init_communicator(device="cpu")

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

        self.dp_rank = get_data_parallel_rank()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.pp_rank = get_pipeline_model_parallel_rank()

        self.fixed_size_obj_collectives = fixed_size_for_obj_collectives

        # Pre-create constant tensors for XLA optimization.
        # These are used in clamp operations and comparisons. Creating them once avoids
        # repeated tensor allocations that cause XLA graph fragmentation.
        device = xm.xla_device()
        self._one_float = torch.tensor(1.0, dtype=torch.float32, device=device)
        self._one_long = torch.tensor(1, dtype=torch.long, device=device)
        self._inf_float = torch.tensor(float("inf"), dtype=torch.float32, device=device)

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        if self.accelerator.num_processes == 1:
            sampler = RepeatSampler(
                data_source=dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
            )
        else:
            trn_config = self.accelerator.state.trn_config
            num_replicas = trn_config.data_parallel_size
            rank = parallel_layers.parallel_state.get_data_parallel_rank()
            sampler = DistributedRepeatSampler(
                dataset=dataset,
                mini_repeat_count=self.num_generations,
                batch_size=self.args.generation_batch_size // self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                shuffle=self.shuffle_dataset,
                seed=self.args.seed,
                num_replicas=num_replicas,
                rank=rank,
            )
        return sampler

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ):
        return NeuronTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        # Average the metrics. Values can be either floats or CPU tensors, so we handle both.
        # Using sum() works for both types; we call .item() only on tensors when computing the average.
        metrics = {}
        for key, val_list in self._metrics[mode].items():
            if len(val_list) == 0:
                continue
            # Convert any tensor values to floats for averaging
            float_vals = [v.item() if isinstance(v, torch.Tensor) else v for v in val_list]
            metrics[key] = sum(float_vals) / len(float_vals)

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}

        # Using the NeuronTrainer log method instead of super().log.
        NeuronTrainer.log(self, logs)

        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def _save_checkpoint(self, model=None, trial=None, metrics=None):
        return NeuronTrainer._save_checkpoint(self)

    def _prepare_inputs(self, inputs: Any) -> dict[str, Any]:
        # Explicitly call GRPOTrainer's _prepare_inputs
        return GRPOTrainer._prepare_inputs(self, inputs)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device

        excluded_keys = {"prompt", "completion", "completion_ids"}
        keys = [key for key in inputs[0] if key not in excluded_keys]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        # Separate model-based vs callable reward functions by index
        model_indices = []
        callable_indices = []
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, torch.nn.Module):
                model_indices.append(i)
            else:
                callable_indices.append(i)

        # Collect results: list of (index, tensor) tuples
        reward_columns = []

        if model_indices:
            # Pre-compute texts once if needed (all models use same text format)
            texts = None
            is_conv = is_conversational(inputs[0])

            if is_conv:
                from trl.data_utils import apply_chat_template

            for i in model_indices:
                reward_func = self.reward_funcs[i]
                reward_processing_class = self.reward_processing_classes[i]

                if is_conv:
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]

                reward_inputs = reward_processing_class(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = NeuronTrainer._prepare_inputs(self, reward_inputs)

                with torch.inference_mode():
                    logits = reward_func(**reward_inputs).logits[:, 0]

                reward_columns.append((i, logits))

        if callable_indices:
            # Use numpy for intermediate storage to avoid Python list overhead
            # and enable efficient single-transfer to XLA device
            num_samples = len(prompts)
            callable_rewards_np = np.empty((len(callable_indices), num_samples), dtype=np.float32)

            for local_idx, global_idx in enumerate(callable_indices):
                reward_func = self.reward_funcs[global_idx]
                output = reward_func(
                    prompts=prompts,
                    completions=completions,
                    completion_ids=completion_ids_list,
                    **reward_kwargs,
                )
                for j, r in enumerate(output):
                    callable_rewards_np[local_idx, j] = r if r is not None else np.nan

            # Single tensor creation and transfer from numpy array
            callable_tensor = torch.from_numpy(callable_rewards_np).to(device=device)

            for local_idx, global_idx in enumerate(callable_indices):
                reward_columns.append((global_idx, callable_tensor[local_idx]))

        # Sort by original index to maintain correct column order
        reward_columns.sort(key=lambda x: x[0])

        # Stack all columns at once instead of indexed assignment in loop
        rewards_per_func = torch.stack([col for _, col in reward_columns], dim=1)

        torch_xla.sync()
        rewards_per_func = self.accelerator.gather(rewards_per_func)
        return rewards_per_func

    def _move_model_to_vllm(self):
        if isinstance(self.model, NeuronPeftModel):
            # Get original (unsharded, untransformed) merged weights for vLLM
            original_weights = get_original_merged_weights_for_vllm(self.model)
            # For now, we only support CPU communicator in Neuron environments.
            # The CPU communicator moves weights to CPU before broadcasting, but to avoid a lot of device -> host moves,
            # we move the weights to CPU here once before broadcasting, and the communicator will just broadcast them.
            original_weights = move_all_tensor_to_cpu(original_weights)
            torch_xla.sync()

            # Send weights to vLLM server (only main process for server mode)
            for name, weight in original_weights.items():
                # Clean up parameter name for vLLM
                name = self._fix_param_name_to_vllm(name)

                # if self.vllm_mode == "server" and self.accelerator.is_main_process:
                #     self.vllm_client.update_named_param(name, weight)
                # elif self.vllm_mode == "colocate":
                #     llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                #     llm_model.load_weights([(name, weight)])
        else:
            for name, param in self.model.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                    self.vllm_client.update_named_param(name, param.data)
                elif self.vllm_mode == "colocate":
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    def _generate_single_turn(self, prompts: list[str], images: list | None):
        if self.state.global_step != getattr(self, "_last_loaded_step", -1):
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Take unique prompts since we have num_generations duplicates
        # Use maybe_apply_chat_template to handle both conversational and simple formats
        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"] for prompt in prompts
        ]
        ordered_set_of_prompts = prompts_text[:: self.num_generations]

        if images is not None:
            ordered_set_of_images = images[:: self.num_generations]
        else:
            ordered_set_of_images = None

        # Generate on main process only, then broadcast to all ranks
        if self.tp_rank == self.pp_rank == 0:
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
        else:
            output = None

        # Broadcast output to all ranks
        trn_config = self.accelerator.state.trn_config
        if trn_config.tensor_parallel_size > 1:
            output = broadcast_object_to_tensor_model_parallel_group(
                output, fixed_size=self.fixed_size_obj_collectives
            )
        if trn_config.pipeline_parallel_size > 1:
            output = broadcast_object_to_pipeline_model_parallel_group(
                output, fixed_size=self.fixed_size_obj_collectives
            )

        # Repeat prompt_ids num_generations times to match completion_ids
        prompt_ids = [ids for ids in output["prompt_ids"] for _ in range(self.num_generations)]
        completion_ids = output["completion_ids"]
        logprobs = output["logprobs"]

        # No forward_kwargs for mock vLLM
        forward_kwargs = {}

        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size: int | None = None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        total_batch_size = input_ids.size(0)
        batch_size = batch_size or total_batch_size  # Chunk inputs into smaller batches to reduce memory peak

        # Ensure input batch size is divisible by `batch_size` to avoid issues with XLA graph compilation.
        if total_batch_size % batch_size != 0:
            raise ValueError(
                f"The input_ids batch size must be divisible by `batch_size`, but got {total_batch_size} and "
                f"{batch_size}."
            )

        num_chunks = total_batch_size // batch_size
        device = input_ids.device

        # Pre-allocate output tensors to avoid list accumulation and repeated concatenation.
        # This creates a single graph for all chunks instead of growing graphs.
        all_logps = torch.empty(total_batch_size, logits_to_keep, dtype=torch.float32, device=device)
        all_entropies = (
            torch.empty(total_batch_size, logits_to_keep, dtype=torch.float32, device=device)
            if compute_entropy
            else None
        )

        # Pre-compute VLM slicing indices if needed (avoids .item() calls inside loop).
        # For VLMs with image_grid_thw, we need to compute pixel_values slicing indices upfront.
        if image_grid_thw is not None and pixel_values is not None:
            rows_per_image = image_grid_thw.prod(dim=-1)
            # num_images is a list of ints, so we can compute cumulative sums on CPU
            cum_imgs = [0]
            for n in num_images:
                cum_imgs.append(cum_imgs[-1] + n)

            # Compute row boundaries for each sample using CPU-computed indices
            rows_per_sample_list = []
            for i in range(len(num_images)):
                start_img = cum_imgs[i]
                end_img = cum_imgs[i + 1]
                rows_per_sample_list.append(rows_per_image[start_img:end_img].sum())
            rows_per_sample = torch.stack(rows_per_sample_list)
            # Compute cumulative row indices on device
            cum_rows = torch.cat(
                [torch.zeros(1, dtype=rows_per_sample.dtype, device=device), rows_per_sample.cumsum(0)]
            )
            # Move to CPU once to get all slice indices (single sync instead of per-chunk)
            torch_xla.sync()
            cum_rows_cpu = cum_rows.cpu().tolist()

        for chunk_idx in range(num_chunks):
            start = chunk_idx * batch_size
            end = start + batch_size

            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}

            if image_grid_thw is not None and pixel_values is not None:
                # Use pre-computed CPU indices to avoid .item() calls
                row_start = int(cum_rows_cpu[start])
                row_end = int(cum_rows_cpu[end])
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                img_start = cum_imgs[start]
                img_end = cum_imgs[end]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]

            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start:end]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start:end]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits

            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs

            # Write directly to pre-allocated tensor instead of list append
            all_logps[start:end] = logps

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies[start:end] = entropies

        torch_xla.sync()

        return all_logps, all_entropies

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
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

        # Convert lists of token IDs to padded tensors using XLA-optimized batch padding.
        # This avoids creating many small tensors and multiple device transfers.
        prompt_ids, prompt_mask = batch_pad_sequences(
            prompt_ids_list,
            target_length=self.max_prompt_length,
            padding_value=self.pad_token_id,
            padding_side="left",
            dtype=torch.long,
            device=device,
        )

        completion_ids, completion_mask = batch_pad_sequences(
            completion_ids_list,
            target_length=self.max_completion_length,
            padding_value=self.pad_token_id,
            padding_side="right",
            dtype=torch.long,
            device=device,
        )

        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps, _ = batch_pad_sequences(
                sampling_per_token_logps_list,
                target_length=self.max_completion_length,
                padding_value=0.0,
                padding_side="right",
                dtype=torch.float32,
                device=device,
            )
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask.
        # Use tensor operations instead of Python list iteration for XLA compatibility.
        if self.mask_truncated_completions:
            # Check if last token is NOT eos or pad (meaning sequence was truncated)
            last_tokens = completion_ids[:, -1]
            # A sequence is NOT truncated if its last token is eos or pad
            is_not_truncated = (last_tokens == self.eos_token_id) | (last_tokens == self.pad_token_id)
            completion_mask = completion_mask * is_not_truncated.unsqueeze(1).long()

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
                    with self.model.disable_adapter():
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
        rewards = (rewards_per_func * self.reward_weights.unsqueeze(0)).nansum(dim=1)

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

        # Use direct comparison instead of torch.zeros_like to avoid tensor allocation
        is_std_zero = std_rewards.abs() < 1e-8
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        metrics = defaultdict(list)
        logs = {}

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i])
            metrics[f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i])
            metrics[f"rewards/{reward_func_name}/std"].append(std_func_rewards)

        metrics["reward"].append(mean_grouped_rewards.mean())
        metrics["reward_std"].append(std_rewards.mean())
        metrics["frac_reward_zero_std"].append(is_std_zero.float().mean())

        # Log prompt and completion texts
        self._logs["prompt"].extend(
            gather_object_from_data_parallel_group(prompts_text, fixed_size=self.fixed_size_obj_collectives)
        )
        self._logs["completion"].extend(
            gather_object_from_data_parallel_group(completions_text, fixed_size=self.fixed_size_obj_collectives)
        )
        logs["rewards"] = {}
        logs["advantages"] = []
        for i, name in enumerate(self.reward_func_names):
            logs["rewards"][name] = rewards_per_func[:, i]
        logs["advantages"] = all_process_advantages

        if images is not None:
            self._logs["images"].extend(
                gather_object_from_data_parallel_group(images, fixed_size=self.fixed_size_obj_collectives)
            )

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            # Original code was:
            # delta = delta[completion_mask.bool()]
            # mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            # max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            # But it is not XLA friendly because it involves dynamic indexing before reduction, so we rewrite it as:
            completion_mask_count = completion_mask.sum()
            delta_masked = delta * completion_mask
            sum_delta = delta_masked.sum()
            mean_delta = sum_delta / (completion_mask_count + 1e-10)
            # We can simply take the max of the masked delta because values in delta are >= 0 (torch.abs).
            max_delta = delta_masked.max()

            metrics["sampling/sampling_logp_difference/mean"].append(self.accelerator.gather(mean_delta).mean())
            metrics["sampling/sampling_logp_difference/max"].append(self.accelerator.gather(max_delta).max())

            # Original code was:
            # flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            # min_importance_sampling_ratio = (
            #     torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            # )
            # mean_importance_sampling_ratio = (
            #     torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            # )
            # max_importance_sampling_ratio = (
            #     torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            # )
            # But it is not XLA friendly because it involves dynamic indexing before reduction, so we rewrite it as:
            # Use pre-created inf constant (cast to proper dtype if needed)
            inf_val = self._inf_float.to(dtype=importance_sampling_ratio.dtype)
            masked_is_ratio_for_min = torch.where(
                completion_mask.bool(),
                importance_sampling_ratio,
                inf_val,
            )
            min_importance_sampling_ratio = masked_is_ratio_for_min.min()
            # importance_sampling_ratio values are >= 0 (torch.exp) so we can use the same computation as for delta.
            flat_is_ratio_masked = importance_sampling_ratio * completion_mask
            sum_flat_is_ratio = flat_is_ratio_masked.sum()
            mean_importance_sampling_ratio = sum_flat_is_ratio / (completion_mask_count + 1e-10)
            max_importance_sampling_ratio = flat_is_ratio_masked.max()

            metrics["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio))
            )
            metrics["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean()
            )
            metrics["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio))
            )

        # Move metrics and logs to CPU. Keep metrics as CPU tensors instead of calling .item()
        # immediately - this defers the sync overhead to when metrics are actually logged.
        torch_xla.sync()
        metrics = move_all_tensor_to_cpu(metrics)
        logs = move_all_tensor_to_cpu(logs)

        # Update the actual metrics and logs.
        self._metrics[mode].update(metrics)
        for name in self.reward_func_names:
            self._logs["rewards"][name].extend(logs["rewards"][name].tolist())
        self._logs["advantages"].extend(logs["advantages"].tolist())

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

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Compute a mask for high-entropy tokens (above the given quantile threshold).

        This XLA-optimized implementation avoids:
        1. Dynamic indexing (sorted_values[quantile_idx]) by using torch.gather
        2. Repeated tensor creation by using pre-created constants
        3. Complex control flow that would cause graph breaks
        """
        pad_value = -1e9
        dtype = entropies.dtype

        # Create pad tensor from pre-allocated constant (avoids allocation in hot path)
        # Note: pad_value is negative, so we can't use self._inf_float directly
        pad_tensor = torch.full_like(entropies[:1, :1], pad_value).expand_as(entropies)

        masked_entropies = torch.where(mask.bool(), entropies, pad_tensor)

        local_flat = masked_entropies.view(-1)
        gathered = self.accelerator.gather(local_flat)

        # Sort gathered values, so that pad_value sentinels are at the beginning
        sorted_values, _ = torch.sort(gathered)

        # Compute the number of valid (non-sentinel) values using a tolerance for float comparison
        is_sentinel = sorted_values < (pad_value + 1e-6)  # pad_value is -1e9
        num_sentinels = is_sentinel.sum()
        num_valid_values = gathered.numel() - num_sentinels

        # Get the quantile index and the corresponding entropy threshold value
        # Use torch.gather instead of dynamic indexing to maintain XLA compatibility
        quantile_idx = num_sentinels + (threshold * num_valid_values.float()).long()
        quantile_idx = quantile_idx.clamp(min=0, max=gathered.numel() - 1)

        # Use gather for XLA-compatible indexing (gather works with tensor indices)
        entropy_threshold = sorted_values.gather(0, quantile_idx.view(1)).squeeze(0)

        # Handle empty case: if everything is sentinel, set threshold to +inf so no token is selected
        has_valid = num_valid_values > 0
        inf_val = self._inf_float.to(dtype=dtype)
        entropy_threshold = torch.where(has_valid, entropy_threshold, inf_val)

        entropy_mask = (entropies > entropy_threshold) & mask.bool()
        return entropy_mask

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

        with self.metrics_collector.time_metric("forward_pass", inputs=inputs):
            # Compute the per_token_logps and the entropy at each position in the completion
            per_token_logps, entropies = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=self.args.per_device_train_batch_size,
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
                # Use pre-created constant instead of creating tensor in hot path
                log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)
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

            # Use scalar min value for clamp instead of creating tensors.
            # PyTorch clamp accepts Python scalars which avoids tensor allocation overhead.
            if self.loss_type == "grpo":
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1)).mean()
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "bnpo":
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1)
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "dr_grpo":
                loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
                loss = loss / self.current_gradient_accumulation_steps
            elif self.loss_type == "dapo":
                normalizer = inputs["num_items_in_batch"]
                loss = (per_token_loss * completion_mask).sum() / normalizer
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            # Log the metrics
            mode = "train" if self.model.training else "eval"

            # Use scalar min value for clamp instead of creating tensor
            completion_token_count = completion_mask.sum().clamp(min=1)

            def masked_batch_mean(x):
                if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                    return x.mean()
                else:
                    return (x * completion_mask).sum() / completion_token_count

            metrics = defaultdict(list)

            if self.beta != 0.0:
                mean_kl = masked_batch_mean(per_token_kl)
                metrics["kl"].append(self.accelerator.gather(mean_kl).nanmean())

            mean_entropy = masked_batch_mean(entropies)
            metrics["entropy"].append(self.accelerator.gather(mean_entropy).nanmean())

            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            metrics["clip_ratio/low_mean"].append(gathered_low_clip.nanmean())
            metrics["clip_ratio/low_min"].append(nanmin(gathered_low_clip))
            gathered_high_clip = self.accelerator.gather(high_clip)
            metrics["clip_ratio/high_mean"].append(gathered_high_clip.nanmean())
            metrics["clip_ratio/high_max"].append(nanmax(gathered_high_clip))
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            metrics["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean())

            # Move metrics to CPU but keep as tensors. The log() method will call .item()
            # when averaging. This defers sync overhead to logging time.
            metrics = move_all_tensor_to_cpu(metrics)
            torch_xla.sync()

            self._metrics[mode].update(metrics)

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
