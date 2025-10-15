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

from typing import Any, Callable

import datasets
import torch
from optimum.utils import logging
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer_callback import TrainerCallback

from ..accelerate import NeuronPartialState
from ..peft import NeuronPeftModel, get_peft_model
from ..utils import (
    is_trl_available,
)
from ..utils.import_utils import is_peft_available
from .sft_config import NeuronSFTConfig
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import SFTConfig, SFTTrainer
else:

    class SFTTrainer:
        pass

    class SFTConfig:
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


class NeuronSFTTrainer(_SFTTrainer):
    """
    `SFTTrainer` adapted for Neuron.

    It differs from the original `SFTTrainer` by:
        - Using `_TrainerForNeuron.__init__()` instead of `Trainer.__init__()`
        - Using the `_TrainerForNeuron.train()` instead of `Trainer.train()`
        - Adapts the `_prepare_non_packed_dataloader` to pad to max length. In the original `SFTTrainer` examples are
          not padded, which is an issue here because it triggers compilation every time.
    """

    def __init__(
        self,
        model: PreTrainedModel | torch.nn.Module | str,
        args: SFTConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
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
        # Deprecated parameters for backward compatibility
        tokenizer: PreTrainedTokenizerBase | None = None,  # Use processing_class instead
    ):
        if not is_trl_available(required_version=TRL_VERSION):
            raise RuntimeError(f"Using NeuronSFTTrainer requires trl=={TRL_VERSION}.")

        from trl.extras.dataset_formatting import get_formatting_func_from_dataset
        from trl.trainer.callbacks import RichProgressCallback
        from trl.trainer.utils import peft_module_casting_to_bf16

        if is_peft_available():
            from peft import PeftConfig

        # Handle backward compatibility for tokenizer parameter
        if tokenizer is not None and processing_class is None:
            processing_class = tokenizer

        args_is_none = args is None
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = NeuronSFTConfig(f"{model_name}-SFT")
        elif args is not None and args.__class__.__name__ == "NeuronTrainingArguments":
            args_as_dict = args.to_dict()
            # Manually copy token values as TrainingArguments.to_dict() redacts them
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = NeuronSFTConfig(**args_as_dict)

        # Set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # We wait for the verbosity of the logger to be set before logging the warning below.
        if args_is_none:
            logging.warning(f"No `SFTConfig` passed, using `output_dir={args.output_dir}`.")

        # Model handling - use model_init_kwargs from args
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `SFTConfig`. Expected either 'auto' or a string representing "
                    f"a valid `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `SFTConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        if is_peft_available() and peft_config is not None:
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the NeuronPeftModel, you need to pass a PeftConfig object to the NeuronSFTTrainer."
                    f" and you passed a {type(peft_config)}."
                )

            if not isinstance(model, NeuronPeftModel):
                gradient_checkpointing_kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
                if getattr(args, "gradient_checkpointing", False) and (
                    "use_reentrant" not in gradient_checkpointing_kwargs
                    or gradient_checkpointing_kwargs["use_reentrant"]
                ):
                    # For backward compatibility with older versions of transformers
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                    else:

                        def make_inputs_require_grad(module, input, output):
                            output.requires_grad_(True)

                        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                model = get_peft_model(model, peft_config)
                if args is not None and args.bf16:
                    peft_module_casting_to_bf16(model)

        # Processing class (tokenizer) handling
        if processing_class is None:
            from transformers import AutoProcessor

            processing_class = AutoProcessor.from_pretrained(model_id)

        # Ensure we have a pad token
        if hasattr(processing_class, "pad_token") and getattr(processing_class, "pad_token", None) is None:
            processing_class.pad_token = processing_class.eos_token

        # Handle max_length (renamed from max_seq_length)
        if args.max_length is None:
            # to overcome some issues with broken tokenizers
            args.max_length = min(processing_class.model_max_length, 1024)

            logger.warning(
                f"You didn't pass a `max_length` argument to the SFTTrainer, this will default to {args.max_length}"
            )

        self.dataset_num_proc = args.dataset_num_proc

        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")

        # Vision Language Model (VLM) support - not yet supported in Neuron
        self._is_vlm = False

        if args.dataset_kwargs is None:
            args.dataset_kwargs = {}

        if formatting_func is None and args.dataset_text_field is None:
            # check if dataset has ChatML format or instruction format and is supported
            # if not stays #None
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)
            # if a template is detected, we don't need to add special tokens again
            if formatting_func is not None:
                args.dataset_kwargs["add_special_tokens"] = False

        if not args.packing:
            # If we aren't skipping data preparation, then a dataset_text_field
            # or formatting_func must be provided.
            if (
                args.dataset_text_field is None
                and formatting_func is None
                and not args.dataset_kwargs.get("skip_prepare_dataset", False)
            ):
                raise ValueError(
                    "You passed `packing=False` to the SFTTrainer/SFTConfig, but you didn't pass a `dataset_text_field` or `formatting_func` argument."
                )

            if data_collator is None:
                data_collator = DataCollatorForLanguageModeling(tokenizer=processing_class, mlm=False)

        # Pre-process the datasets only once per node. The remaining processes will use the cache.
        with NeuronPartialState().local_main_process_first():
            if train_dataset is not None:
                train_dataset = self._prepare_dataset(
                    train_dataset, processing_class, args, args.packing, formatting_func, "train"
                )
            if eval_dataset is not None:
                _multiple = isinstance(eval_dataset, dict)
                _eval_datasets = eval_dataset if _multiple else {"singleton": eval_dataset}

                for _eval_dataset_name, _eval_dataset in _eval_datasets.items():
                    _eval_datasets[_eval_dataset_name] = self._prepare_dataset(
                        _eval_dataset,
                        processing_class,
                        args,
                        args.eval_packing if args.eval_packing is not None else args.packing,
                        formatting_func,
                        _eval_dataset_name,
                    )
                if not _multiple:
                    eval_dataset = _eval_datasets["singleton"]

        if (
            hasattr(processing_class, "padding_side")
            and processing_class.padding_side is not None
            and processing_class.padding_side != "right"
        ):
            logger.warning(
                "You passed a processing_class with `padding_side` not equal to `right` to the SFTTrainer. This might lead to some unexpected behaviour due to "
                'overflow issues when training a model in half-precision. You might consider adding `processing_class.padding_side = "right"` to your code.'
            )

        NeuronTrainer.__init__(
            self,
            model,
            args,
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

        if self.args.max_steps > 0 and args.packing:
            logger.warning(
                "You passed `packing=True` to the NeuronSFTTrainer/SFTConfig, and you are training your model with `max_steps` strategy. The dataset will be iterated until the `max_steps` are reached."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and args.packing:
            self.train_dataset.infinite = False

        if any(isinstance(callback, RichProgressCallback) for callback in self.callback_handler.callbacks):
            for callback in self.callback_handler.callbacks:
                # Remove the PrinterCallback to avoid duplicated prints in case we passed a `RichProgressCallback`
                if callback.__class__.__name__ == "PrinterCallback":
                    self.callback_handler.pop_callback(callback)

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ):
        return NeuronTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint)

    def _prepare_dataset(
        self,
        dataset,
        processing_class,
        args,
        packing,
        formatting_func=None,
        dataset_name="train",
    ):
        """
        Prepare dataset for Neuron training with proper padding.

        This method overrides the base TRL implementation to ensure consistent padding
        for Neuron devices, which require fixed input shapes to avoid recompilation.
        """
        # For packing, delegate to parent implementation but ensure no padding_free
        if packing:
            # Temporarily disable padding_free for packing as well
            original_padding_free = getattr(args, "padding_free", False)
            args.padding_free = False
            try:
                result = super()._prepare_dataset(
                    dataset, processing_class, args, packing, formatting_func, dataset_name
                )
            finally:
                args.padding_free = original_padding_free
            return result

        # For non-packed datasets, use our custom implementation with forced padding
        from datasets import Dataset

        # Apply formatting function if provided
        if formatting_func is not None:
            if isinstance(dataset, Dataset):
                dataset = dataset.map(
                    lambda example: {"text": formatting_func(example)},
                    num_proc=args.dataset_num_proc,
                    desc=f"Applying formatting function to {dataset_name} dataset",
                )
            else:  # IterableDataset
                dataset = dataset.map(lambda example: {"text": formatting_func(example)})

        # Tokenization function with forced padding for Neuron
        def tokenize(examples):
            # Handle both single examples and batches
            if isinstance(examples[args.dataset_text_field], list):
                texts = examples[args.dataset_text_field]
            else:
                texts = [examples[args.dataset_text_field]]

            outputs = processing_class(
                texts,
                add_special_tokens=True,
                truncation=True,
                # Critical for Neuron: always pad to max_length to avoid recompilation
                padding="max_length",
                max_length=args.max_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            return {
                "input_ids": outputs["input_ids"],
                "attention_mask": outputs["attention_mask"],
                "labels": outputs["input_ids"].copy(),  # For language modeling
            }

        # Build map kwargs
        map_kwargs = {
            "batched": True,
            "remove_columns": dataset.column_names
            if hasattr(dataset, "column_names") and dataset.column_names
            else None,
        }
        if isinstance(dataset, Dataset):
            map_kwargs["num_proc"] = args.dataset_num_proc
            map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

        tokenized_dataset = dataset.map(tokenize, **map_kwargs)
        return tokenized_dataset
