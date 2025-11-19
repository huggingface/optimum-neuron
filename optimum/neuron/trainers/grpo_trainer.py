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
import torch_xla
import torch_xla.core.xla_model as xm
from accelerate.utils import set_seed
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from optimum.utils import logging
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_rich_available

from ..models.training import NeuronModelForCausalLM
from ..peft import NeuronPeftModel, get_peft_model
from ..utils import is_precompilation, is_trl_available
from ..utils.import_utils import is_peft_available
from .grpo_config import NeuronGRPOConfig
from .training_args import NeuronTrainingArguments
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION, nanmax, nanmin, nanstd, neuron_parallel_compile_tokenizer_decoder_method, pad


if is_wandb_available():
    import wandb

if is_trl_available():
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import is_conversational
    from trl.trainer.utils import (
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
        return NeuronTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

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
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            if isinstance(reward_func, torch.nn.Module):
                if is_conversational(inputs[0]):
                    from trl.data_utils import apply_chat_template
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = NeuronTrainer._prepare_inputs(self, reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                torch_xla.sync()
            else:
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                )
                output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = self.accelerator.gather(rewards_per_func)
        torch_xla.sync()
        return rewards_per_func

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

    def _to_fixed_length(
        self,
        tensor: torch.Tensor,
        padding_value: int = 0,
        padding_side: str = "right"
    ) -> torch.Tensor:
        """
        Pads or truncates tensor to fixed length = max_prompt_length + max_completion_length.
        """
        fixed_length = self.max_prompt_length + self.max_completion_length
        seq_len = tensor.shape[1]

        if seq_len == fixed_length:
            return tensor
        elif seq_len < fixed_length:
            # Pad to fixed length
            pad_amount = fixed_length - seq_len
            pad_config = (pad_amount, 0) if padding_side == "left" else (0, pad_amount)
            return torch.nn.functional.pad(tensor, pad_config, value=padding_value)
        else:
            # Truncate to fixed length
            if padding_side == "left":
                return tensor[:, -fixed_length:]
            else:
                return tensor[:, :fixed_length]

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size, # Compared to the original `trl` implementation, `batch_size` must be specified.
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Make sure the inputs have a fixed shape.
        input_ids = self._to_fixed_length(
            input_ids, padding_value=self.pad_token_id, padding_side="left"
        )
        attention_mask = self._to_fixed_length(
            attention_mask, padding_value=0, padding_side="left"
        )

        # Force synchronization before starting computation to re-use the same graph.
        torch_xla.sync()

        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        # TODO: check if it's ok with TORCH XLA
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                # TODO: not support with torch XLA, fix it later.
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

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
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

            torch_xla.sync()

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None

        # Force synchronization after computation to ensure graph is re-used.
        torch_xla.sync()

        return logps, entropies

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
        # self._logs["prompt"].extend(self.accelerator.gather_object(prompts_text))
        # self._logs["completion"].extend(self.accelerator.gather_object(completions_text))
        logs["rewards"] = {}
        logs["advantages"] = []
        for i, name in enumerate(self.reward_func_names):
            logs["rewards"][name] = rewards_per_func[:, i]
        logs["advantages"] = all_process_advantages

        # if images is not None:
        #     self._logs["images"].extend(self.accelerator.gather_object(images))

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

            metrics["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean()
            )
            metrics["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max()
            )

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
            masked_is_ratio_for_min = torch.where(
                completion_mask.bool(),
                importance_sampling_ratio,
                torch.tensor(float('inf'), device=device, dtype=importance_sampling_ratio.dtype)
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

        # Graph break after metrics and logs computation.
        torch_xla.sync()

        # Move metrics and logs to CPU.
        metrics = move_all_tensor_to_cpu(metrics)
        metrics = {key: [val.item() for val in value] for key, value in metrics.items()}
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
        # Original code does the following:
        # local = entropies[mask.bool()].float()
        # # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # # This guarantees that the sentinel cannot collide with any real entropy value.
        # pad_value = -1e9

        # # Pad across processes so that every rank has the same tensor length
        # padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        # gathered = self.accelerator.gather(padded)

        # # Drop sentinel values (safe because no entropy can be negative)
        # gathered = gathered[gathered != pad_value]

        # if gathered.numel() == 0:
        #     return torch.zeros_like(entropies, dtype=torch.bool)

        # entropy_threshold = torch.quantile(gathered, threshold)
        # masked_entropies = entropies * mask.float()
        # entropy_mask = masked_entropies >= entropy_threshold
        # return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

        pad_value = -1e9
        device = entropies.device

        masked_entropies = torch.where(
            mask.bool(),
            entropies,
            torch.tensor(pad_value, device=device, dtype=entropies.dtype),
        )

        local_flat = masked_entropies.view(-1)
        gathered = self.accelerator.gather(local_flat)

        # Sort gathered values, so that pad_value sentinels are at the beginning
        sorted_values, _ = torch.sort(gathered)

        # Compute the number of valid (non-sentinel) values
        num_valid = (sorted_values != pad_value).sum()
        num_sentinels = (sorted_values == pad_value).sum()
        valid_start_idx = num_sentinels
        num_valid_values = gathered.numel() - num_sentinels

        # Get the quantile index and the corresponding entropy threshold value
        quantile_idx = valid_start_idx + (threshold * num_valid_values).long()
        quantile_idx = quantile_idx.clamp(max=gathered.numel() - 1)
        entropy_threshold = sorted_values[quantile_idx]

        # Handle empty case, if everything is sentinel, set threshold to +inf so no token is selected
        has_valid = num_valid > 0
        entropy_threshold = torch.where(
            has_valid,
            entropy_threshold,
            torch.tensor(float('inf'), device=device, dtype=entropies.dtype)
        )

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
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
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

        torch_xla.sync()  # Graph break before moving metrics to CPU.
        metrics = move_all_tensor_to_cpu(metrics)
        metrics = {key: [val.item() for val in value] for key, value in metrics.items()}

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
