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

import os
from typing import Any, Callable

import datasets
import torch
from optimum.utils import logging
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback

from ..models.training import NeuronModelForCausalLM
from ..peft import NeuronPeftModel, get_peft_model
from ..utils import (
    is_trl_available,
)
from ..utils.import_utils import is_peft_available
from .sft_config import NeuronSFTConfig
from .training_args import NeuronTrainingArguments
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import SFTConfig, SFTTrainer
    from trl.models import clone_chat_template
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling, DataCollatorForVisionLanguageModeling
else:

    class SFTTrainer:
        pass

    class SFTConfig:
        pass

    class DataCollatorForLanguageModeling:
        pass

    class DataCollatorForVisionLanguageModeling:
        pass

    def clone_chat_template(*args, **kwargs):
        pass


if is_peft_available():
    from peft import PeftConfig
else:

    class PeftConfig:
        pass


# Create a new class that inherits from NeuronTrainer to use this class instead of the transformers Trainer,
# but has the same methods and attributes as SFTTrainer.
# We can then inherit from this class to create our NeuronSFTTrainer.
_SFTTrainer = type(
    "_SFTTrainer",
    (NeuronTrainer,),
    SFTTrainer.__dict__.copy(),
)


logger = logging.get_logger()


class NeuronDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Data collator for Neuron that ensures all sequences are padded to exactly max_length.

    This is required for Neuron devices to maintain fixed input shapes and avoid recompilation.
    Inherits from trl's DataCollatorForLanguageModeling but adds max_length enforcement.
    """

    def __init__(self, max_length: int, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length

    def __call__(self, examples):
        # Pad/truncate all sequences to max_length before calling parent
        for example in examples:
            if "input_ids" in example:
                input_ids = example["input_ids"]
                current_length = len(input_ids)

                if current_length > self.max_length:
                    # Truncate to max_length
                    example["input_ids"] = input_ids[: self.max_length]
                elif current_length < self.max_length:
                    # Pad to max_length
                    example["input_ids"] = input_ids + [self.pad_token_id] * (self.max_length - current_length)

                # Handle other fields if present
                for key in ["labels", "attention_mask", "completion_mask"]:
                    if key in example:
                        field = example[key]
                        field_length = len(field)
                        if field_length > self.max_length:
                            example[key] = field[: self.max_length]
                        elif field_length < self.max_length:
                            # Pad with appropriate value
                            if key == "labels":
                                pad_value = -100
                            elif key == "attention_mask":
                                pad_value = 0
                            elif key == "completion_mask":
                                pad_value = 0
                            example[key] = field + [pad_value] * (self.max_length - field_length)

        return super().__call__(examples)


class NeuronSFTTrainer(_SFTTrainer):
    """
    `SFTTrainer` adapted for Neuron (Trainium) devices.
    """

    def __init__(
        self,
        model: PreTrainedModel | torch.nn.Module | str,
        args: SFTConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: "Dataset | IterableDataset | datasets.Dataset | None" = None,
        eval_dataset: "Dataset | dict[str, Dataset] | datasets.Dataset | None" = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: PeftConfig | None = None,
        formatting_func: Callable | None = None,
    ):
        if not is_trl_available(required_version=TRL_VERSION):
            raise RuntimeError(f"Using NeuronSFTTrainer requires trl=={TRL_VERSION}.")

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = NeuronSFTConfig(f"{model_name}-SFT")
        elif isinstance(args, NeuronTrainingArguments) and not isinstance(args, NeuronSFTConfig):
            dict_args = args.to_dict()
            dict_args["hub_token"] = args.hub_token
            dict_args.pop("push_to_hub_token", None)
            args = NeuronSFTConfig(**dict_args)

        # Model
        if isinstance(model, str):
            model = NeuronModelForCausalLM.from_pretrained(model, **args.model_init_kwargs or {})
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `SFTConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )
        model_id = model.config._name_or_path

        # Processing class
        if processing_class is None:
            from transformers import AutoProcessor

            processing_class = AutoProcessor.from_pretrained(model_id)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
            self._is_vlm = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
            self._is_vlm = False
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if args.eos_token is not None:
            eos_token = args.eos_token
            eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token_id is None:
                raise ValueError(
                    f"The specified `eos_token` ('{eos_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `eos_token` exists "
                    "in the vocabulary before using it as an EOS token."
                )
            tokenizer.eos_token_id = eos_token_id

        if args.chat_template_path is not None:
            if os.path.isfile(args.chat_template_path) and args.chat_template_path.endswith((".jinja", ".j2")):
                with open(args.chat_template_path, encoding="utf-8") as chat_template_file:
                    processing_class.chat_template = chat_template_file.read()
                added_tokens = []
            else:
                model, processing_class, added_tokens = clone_chat_template(
                    model, processing_class, args.chat_template_path
                )
        else:
            added_tokens = []

        # Catch some wrong configurations related to VLMs
        if self._is_vlm and args.packing:
            raise ValueError(
                "Packing is not supported for vision-language models. Please set `packing=False` in the SFTConfig."
            )
        if self._is_vlm and args.padding_free:
            raise ValueError(
                "Padding-free training is yet not supported for vision-language models. Please set "
                "`padding_free=False` in the `SFTConfig`."
            )

        # PEFT configuration and model wrapping
        if peft_config is not None:
            if added_tokens:
                # Ensure that the added tokens are trainable
                if peft_config.trainable_token_indices is None:
                    peft_config.trainable_token_indices = {"embed_tokens": added_tokens}
                elif "embed_tokens" not in peft_config.trainable_token_indices:
                    peft_config.trainable_token_indices["embed_tokens"] = added_tokens
                else:
                    peft_config.trainable_token_indices["embed_tokens"].extend(added_tokens)

                # Ensure that the lm_head is trainable
                if peft_config.modules_to_save is None or "lm_head" not in peft_config.modules_to_save:
                    logger.warning(
                        "Cloning chat template added new tokens to the tokenizer, but 'lm_head' is not in PEFT's "
                        "`modules_to_save`. As a result, the model may not learn to generate outputs with these new "
                        "tokens, leading to degraded generation quality. To fix this, add "
                        "`modules_to_save=['lm_head']` to your PEFT configuration."
                    )

                    if peft_config.modules_to_save is None:
                        peft_config.modules_to_save = ["lm_head"]
                    else:
                        peft_config.modules_to_save.append("lm_head")

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

        # Data collator
        # Neuron-specific: padding_free must always be False for Neuron devices
        self.padding_free = False
        if args.padding_free:
            logger.warning(
                "padding_free=True is not supported for Neuron training. Neuron devices require fixed input shapes "
                "to avoid recompilation. Setting padding_free=False."
            )
            args.padding_free = False

        # Decide whether to use completion-only loss
        dataset_sample = next(iter(train_dataset))
        if args.completion_only_loss is None:
            self.completion_only_loss = "prompt" in dataset_sample and "completion" in dataset_sample
        else:
            self.completion_only_loss = args.completion_only_loss

        self._is_vision_dataset = "image" in dataset_sample or "images" in dataset_sample
        if self._is_vision_dataset and not self._is_vlm:
            raise ValueError(
                "The dataset appears to be vision-related (contains 'image' or 'images' keys), but the provided "
                "model does not seem to be a vision-language model. Please check your model and dataset."
            )

        if data_collator is None and not self._is_vision_dataset:
            # Get the pad token
            pad_token = args.pad_token or tokenizer.pad_token or tokenizer.eos_token
            pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)
            if pad_token_id is None:
                raise ValueError(
                    f"The specified `pad_token` ('{pad_token}') is not found in the vocabulary of the given "
                    f"`processing_class` ({processing_class.__class__.__name__}). Ensure that the `pad_token` exists "
                    "in the vocabulary before using it as a padding token."
                )
            # Neuron-specific: use NeuronDataCollatorForLanguageModeling to ensure fixed max_length padding
            data_collator = NeuronDataCollatorForLanguageModeling(
                max_length=args.max_length,
                pad_token_id=pad_token_id,
                completion_only_loss=self.completion_only_loss,
                padding_free=self.padding_free,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )
        elif data_collator is None and self._is_vision_dataset:
            data_collator = DataCollatorForVisionLanguageModeling(
                processor=processing_class,
                max_length=args.max_length,
                completion_only_loss=self.completion_only_loss,
                pad_to_multiple_of=args.pad_to_multiple_of,
                dataset_text_field=args.dataset_text_field,
            )

        # Dataset
        skip_prepare_dataset = (
            args.dataset_kwargs is not None
            and args.dataset_kwargs.get("skip_prepare_dataset", False)
            or self._is_vision_dataset
        )
        if not skip_prepare_dataset:
            if self.completion_only_loss and formatting_func:
                raise ValueError(
                    "A formatting function was provided while `completion_only_loss=True`, which is incompatible. "
                    "Using a formatter converts the dataset to a language modeling type, conflicting with "
                    "completion-only loss. To resolve this, apply your formatting function before passing the "
                    "dataset, or disable `completion_only_loss` in `SFTConfig`."
                )
            train_dataset = self._prepare_dataset(
                train_dataset, processing_class, args, args.packing, formatting_func, "train"
            )
            if eval_dataset is not None:
                packing = args.packing if args.eval_packing is None else args.eval_packing
                if isinstance(eval_dataset, dict):
                    eval_dataset = {
                        key: self._prepare_dataset(dataset, processing_class, args, packing, formatting_func, key)
                        for key, dataset in eval_dataset.items()
                    }
                else:
                    eval_dataset = self._prepare_dataset(
                        eval_dataset, processing_class, args, packing, formatting_func, "eval"
                    )

        # Neuron-specific: we don't support NeFTune
        self._trainer_supports_neftune = False

        # Initialize NeuronTrainer
        NeuronTrainer.__init__(
            self,
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ):
        return NeuronTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint)

    def log(self, logs: dict[str, float]) -> None:
        """
        Override SFTTrainer's log method to use NeuronTrainer's implementation.

        SFTTrainer has custom metrics tracking that we don't use for Neuron training.
        """
        return NeuronTrainer.log(self, logs)

    def _save_checkpoint(self, model=None, trial=None, metrics=None):
        """
        Override SFTTrainer's _save_checkpoint to use NeuronTrainer's implementation.

        SFTTrainer has a custom checkpoint saving method, but we use NeuronTrainer's
        which is compatible with Neuron's distributed training and async saving.
        NeuronTrainer._save_checkpoint only takes self, so we ignore the extra arguments.
        """
        return NeuronTrainer._save_checkpoint(self)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for Neuron-optimized training.

        Overrides TRL SFTTrainer's compute_loss to set use_cache=False for gradient
        checkpointing compatibility and delegate to NeuronTrainer's compute_loss.
        """
        # Set use_cache to False to avoid warnings with gradient checkpointing
        inputs["use_cache"] = False

        # Call the parent NeuronTrainer's compute_loss method (not TRL's)
        return NeuronTrainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)

    def training_step(
        self, model: torch.nn.Module, inputs: dict[str, Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step for Neuron-optimized training.

        Overrides SFTTrainer.training_step to delegate to NeuronTrainer's implementation,
        which is compatible with Neuron's distributed training setup.
        """
        return NeuronTrainer.training_step(self, model, inputs, num_items_in_batch=num_items_in_batch)
