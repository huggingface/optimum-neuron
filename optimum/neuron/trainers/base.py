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

import contextlib
import functools
from functools import partial
import inspect
import math
import os
import shutil
import sys
import time
import warnings
from typing import Any, Callable, Type, TYPE_CHECKING

import numpy as np
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_replica_groups,
    get_data_parallel_size,
    model_parallel_is_initialized,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
)
import torch
import torch.nn as nn
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.runtime as xr
from torch.utils.data import IterableDataset
from accelerate import __version__ as accelerate_version
from accelerate.utils import AutocastKwargs, DataLoaderConfiguration
from neuronx_distributed.pipeline import NxDPPModel
from packaging import version
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.training_args import OptimizerNames
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import is_datasets_available
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.image_processing_utils import BaseImageProcessor
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.processing_utils import ProcessorMixin
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.integrations import hp_params, get_reporting_integration_callbacks
from accelerate.data_loader import SeedableRandomSampler
from transformers.modeling_utils import unwrap_model
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    get_parameter_names,
    LabelSmoother,
    find_batch_size,
    get_dataloader_sampler,
    nested_concat,
    nested_numpify,
    reissue_pt_warnings,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    seed_worker,
    get_last_checkpoint,
    EvalLoopOutput,
    RemoveColumnsCollator,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    SaveStrategy,
    TrainOutput,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.utils import (
    WEIGHTS_NAME,
    is_accelerate_available,
    find_labels,
    can_return_loss,
)
from transformers.optimization import get_scheduler
from transformers.models.auto.modeling_auto import MODEL_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from optimum.utils import logging

from ..accelerate import NeuronAccelerator, NeuronDistributedType
from ..cache.hub_cache import hub_neuronx_cache, synchronize_hub_cache
from ..cache.training import patch_neuron_cc_wrapper
from ..utils import (
    patch_within_function,
)
from ..utils.cache_utils import (
    get_hf_hub_cache_repos,
    get_neuron_cache_path,
)
from ..peft import NeuronPeftModel
from ..utils.misc import is_main_worker, is_precompilation
from ..utils.require_utils import requires_torch_neuronx
from ..utils.training_utils import (
    get_model_param_count,
    is_logging_process,
    is_logging_process_method,
    patch_generation_mixin_to_neuron_generation_mixin,
    skip_first_batches,
)
from .training_args import NeuronTrainingArguments


logger = logging.get_logger()

if is_datasets_available():
    import datasets

TRL_VERSION = "0.11.4"

KEEP_HF_HUB_PROGRESS_BARS = os.environ.get("KEEP_HF_HUB_PROGRESS_BARS")
if KEEP_HF_HUB_PROGRESS_BARS is None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

transformers_get_optimizer_cls_and_kwargs = Trainer.get_optimizer_cls_and_kwargs

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCALER_NAME = "scaler.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
FSDP_MODEL_NAME = "pytorch_model_fsdp"


class NeuronTrainer:
    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        args: NeuronTrainingArguments,
        data_collator: DataCollator | None = None,
        train_dataset: "Dataset | IterableDataset | datasets.Dataset | None" = None,
        eval_dataset: "Dataset | dict[str, Dataset] | datasets.Dataset | None" = None,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        # TODO: filter the args that do not make sense for us and raise error for unsupported features.

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = NeuronTrainingArguments(output_dir=output_dir)

        self.args = args
        self.trn_config = self.args.trn_config

        # Distributed training useful attributes
        self.dp_size = self.trn_config.data_parallel_size
        self.tp_size = self.trn_config.tensor_parallel_size
        self.pp_size = self.trn_config.pipeline_parallel_size
        self.pp_rank = get_pipeline_model_parallel_rank()

        self.compute_loss_func = compute_loss_func

        # Set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # TODO: set seed here?

        if model is None:
            raise ValueError("A model must be provided to the Trainer.")

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        self.model = model
        self.create_accelerator_and_postprocess()

        if self.args.use_liger_kernel:
            raise RuntimeError(f"Liger kernel is not supported in NeuronTrainer.")

        default_collator = (
            DataCollatorWithPadding(processing_class)
            if processing_class is not None
            and isinstance(processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
            else default_data_collator
        )
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class

        model_forward = (
            model.forward
            if not isinstance(model, NeuronPeftModel)
            else model.get_base_model().forward
        )
        forward_params = inspect.signature(model_forward).parameters

        # Check if the model has explicit setup for loss kwargs,
        # if not, check if `**kwargs` are in model.forward
        if hasattr(model, "accepts_loss_kwargs"):
            self.model_accepts_loss_kwargs = model.accepts_loss_kwargs
        else:
            self.model_accepts_loss_kwargs = any(
                k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values()
            )

        self.compute_metrics = compute_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs

        if self.optimizer_cls_and_kwargs is not None and self.optimizer is not None:
            raise RuntimeError("Passing both `optimizers` and `optimizer_cls_and_kwargs` arguments is incompatible.")

        if self.optimizer is not None:
            model_device = optimizer_device = None
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )

        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise ValueError("The `data_collator` should be a simple callable (function, class with `__call__`).")

        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        if train_dataset is not None and not has_length(train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        self._signature_columns = None

        # Label smoothing
        if self.args.label_smoothing_factor != 0:
            self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        self.control = TrainerControl()

        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        if isinstance(self.model, NeuronPeftModel) and self.args.label_names is None:
            logger.warning(
                f"No label_names provided for model class `{self.model.__class__.__name__}`."
                " Since `NeuronPeftModel` hides base models input arguments, if label_names is not given, label_names "
                "can't be set automatically within `NeuronTrainer`."
                " Note that empty label_names list will be used instead."
            )
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)


    @property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        logger.warning(
            "NeuronTrainer.tokenizer is now deprecated. You should use NeuronTrainer.processing_class instead."
        )
        return self.processing_class

    def create_accelerator_and_postprocess(self):
        # We explicitly don't rely on the `Accelerator` to do gradient accumulation
        grad_acc_kwargs = {}
        if self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                # raise because we do not know which setting is intended.
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]

        accelerator_config = self.args.accelerator_config.to_dict()

        # TODO: do we support these configurations in the distributed setting?
        dataloader_config = DataLoaderConfiguration(
            split_batches=accelerator_config.pop("split_batches"),
            dispatch_batches=accelerator_config.pop("dispatch_batches"),
            even_batches=accelerator_config.pop("even_batches"),
            use_seedable_sampler=accelerator_config.pop("use_seedable_sampler"),
        )

        args = {
            "deepspeed_plugin": None, # We don't use deepspeed plugin in NeuronTrainer
            "dataloader_config": dataloader_config,
        }

        # create accelerator object
        self.accelerator = NeuronAccelerator(
            *args,
            trn_config=self.trn_config,
            zero_1=self.args.zero_1,
            mixed_precision="bf16" if self.args.bf16 and self.args.use_autocast else "no",
        )

        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

    def add_callback(self, callback: Type[TrainerCallback] | TrainerCallback):
        """
        Add a callback to the current list of `TrainerCallback`.

        Args:
           callback (`Type[TrainerCallback] | TrainerCallback`):
               A `TrainerCallback` class or an instance of a `TrainerCallback`. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback: Type[TrainerCallback] | TrainerCallback) -> TrainerCallback | None:
        """
        Remove a callback from the current list of `TrainerCallback` and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`Type[TrainerCallback] | TrainerCallback`):
               A `TrainerCallback` class or an instance of a `TrainerCallback`. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            `TrainerCallback | None`: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback: Type[TrainerCallback] | TrainerCallback):
        """
        Remove a callback from the current list of `TrainerCallback`.

        Args:
           callback (`Type[TrainerCallback] | TrainerCallback`):
               A `TrainerCallback` class or an instance of a `TrainerCallback`. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            model_to_inspect = self.model
            if isinstance(self.model, NeuronPeftModel):
                if hasattr(self.model, "get_base_model"):
                    model_to_inspect = self.model.get_base_model()
                else:
                    # PeftMixedModel do not provide a `get_base_model` method
                    model_to_inspect = self.model.base_model.model
            signature = inspect.signature(model_to_inspect.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: str | None = None):
        if not is_datasets_available() or not self.args.remove_unused_columns:
            return dataset

        import datasets

        self._set_signature_columns_if_needed()
        # At this point self._signature_columns is guaranteed to be not None
        signature_columns: list = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                "No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self, data_collator: Callable, description: str | None = None
    ) -> Callable:
        """Wrap the data collator in a callable removing unused columns."""
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        remove_columns_collator = RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=signature_columns,
            logger=logger,
            description=description,
            model_name=self.model.__class__.__name__,
        )
        return remove_columns_collator

    def _get_train_sampler(self, train_dataset: Dataset | None = None) -> torch.utils.data.Sampler | None:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None
        # TODO: should we use DistributedSampler in distributed mode or let the accelerator handle the work?
        return RandomSampler(train_dataset)

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Callable[[Dataset], torch.utils.data.Sampler] | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> DataLoader:
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
                dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)


        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,
            "persistent_workers": False,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = DataLoader(dataset, **dataloader_params)
        
        # The NeuronAccelerator will take care of preparing the dataloader, transforming the sampler for distributed
        # training.
        return self.accelerator.prepare_data_loader(dataloader, use_mp_device_loader=True)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self.args.per_device_train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def get_eval_dataloader(self) -> DataLoader:
        raise NotImplementedError("Evaluation is not supported in NeuronTrainer.")

    def get_test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Testing is not supported in NeuronTrainer.")

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

    def get_decay_parameter_names(self, model) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', 'layernorm', or 'rmsnorm')
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS, ["bias", "layernorm", "rmsnorm"])
        return decay_parameters

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if isinstance(self.model, NxDPPModel):
            opt_model = self.model.original_torch_module
            named_parameters = list(self.model.local_named_parameters())
        else:
            opt_model = self.model
            named_parameters = list(self.model.named_parameters())

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in named_parameters if (n in decay_parameters and p.requires_grad)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in named_parameters if (n not in decay_parameters and p.requires_grad)],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def get_num_trainable_parameters(self):
        """
        Get the number of trainable parameters.
        """
        return get_model_param_count(self.model, trainable_only=True)

    def get_learning_rates(self):
        """
        Returns the learning rate of each parameter from self.optimizer.
        """
        if self.optimizer is None:
            raise ValueError("Trainer optimizer is None, please make sure you have setup the optimizer before.")
        return [group["lr"] for group in self.optimizer.param_groups]

    def get_optimizer_group(self, param: str | torch.nn.parameter.Parameter | None = None):
        """
        Returns optimizer group for a parameter if given, else returns all optimizer groups for params.

        Args:
            param (`str | torch.nn.parameter.Parameter | None`, defaults to `None`):
                The parameter for which optimizer group needs to be returned.
        """
        if self.optimizer is None:
            raise ValueError("Trainer optimizer is None, please make sure you have setup the optimizer before.")
        if param is not None:
            for group in self.optimizer.param_groups:
                if param in group["params"]:
                    return group
        return [group["params"] for group in self.optimizer.param_groups]

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments, model: PreTrainedModel | None = None
    ) -> tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAMW_TORCH:
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim == OptimizerNames.SGD:
            optimizer_cls = torch.optim.SGD
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.

        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.train_batch_size

    @staticmethod
    def num_tokens(train_dl: DataLoader, max_steps: int | None = None) -> int:
        """
        Helper to get number of tokens in a [`~torch.utils.data.DataLoader`] by enumerating dataloader.
        """
        train_tokens = 0
        dp_size = get_data_parallel_size()
        try:
            for batch in train_dl:
                tokens = batch["input_ids"].numel()
                if max_steps is not None:
                    return tokens * max_steps * dp_size
                train_tokens += tokens
        except KeyError:
            NeuronTrainer.warning("Cannot get num_tokens from dataloader")
        train_tokens = train_tokens * dp_size
        return train_tokens

    def autocast_smart_context_manager(self, cache_enabled: bool | None = True):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        autocast_handler = AutocastKwargs(
            enabled=self.accelerator.autocast_handler.enabled,
            cache_enabled=cache_enabled,
        )
        return self.accelerator.autocast(autocast_handler=autocast_handler)

    def set_initial_training_values(
        self, args: NeuronTrainingArguments, dataloader: DataLoader, total_train_batch_size: int
    ):
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `epoch_based`
        - `len_dataloader`
        - `max_steps`
        """
        # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None

        # Case 2: We have a dataloader length and can extrapolate
        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        if len_dataloader:
            num_examples = self.num_examples(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        )

    def setup_training(self, train_dataloader: DataLoader, max_steps: int, num_train_epochs: int, num_examples: int, total_train_batch_size: int):
        """
        Setup everything to prepare for the training loop.
        This methods does not return anything but initializes many attributes of the class for training.
        """
        args = self.args

        # Initialize the Trainer state
        self.state = TrainerState()

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        # It is handled differently if pipeline parallelism is enabled.
        if args.gradient_checkpointing and args.pipeline_parallel_size == 1:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        self.model = self.accelerator.prepare_model(self.model, full_bf16=args.bf16)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        if not isinstance(self.model, NxDPPModel):
            self.model.train()

        if hasattr(self.lr_scheduler, "step"):
            self.optimizer = self.accelerator.prepare(self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)

        # Train!
        parameter_count = get_model_param_count(self.model, trainable_only=True)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Data Parallel Size: {args.trn_config.data_parallel_size}")
        logger.info(f"  Tensor Parallel Size: {args.trn_config.tensor_parallel_size}")
        logger.info(f"  Pipeline Parallel Size: {args.trn_config.pipeline_parallel_size}")
        logger.info(f"  Instantaneous batch size per data parallel rank = {args.per_device_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Num trainable parameters = {parameter_count:,}")

        self.state.epoch = 0

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader

        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs

        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = is_logging_process()

        self.global_step_last_logged = 0

        self.optimizer.zero_grad()
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        dtype = torch.bfloat16 if args.bf16 else torch.float32
        self.loss = torch.tensor(0.0).to(dtype).to(xm.xla_device())
        self.grad_norm = None

    def train_step(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        manager = self.autocast_smart_context_manager()
        pp_size = self.trn_config.pipeline_parallel_size

        # We need to move the inputs to the device only if we are not using pipeline parallelism because the
        # NxDPPModel class handles the device placement of the inputs.
        if pp_size == 1:
            inputs = tree_map(lambda t: t.to(xm.xla_device()) if isinstance(t, torch.Tensor) else t, inputs)

        if isinstance(model, NxDPPModel):
            with manager:
                loss = model.run_train(**inputs)

            # When using pipeline parallelism, the loss is only computed on the last stage.
            # So we set the loss to zero on other stages.
            if self.pp_rank != self.pp_size - 1:
                dtype = torch.bfloat16 if self.args.bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
        else:
            if self.model_accepts_loss_kwargs and "labels" in inputs:
                inputs = dict(**inputs, reduction="sum")

            with manager:
                outputs = model(**inputs)

            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss

    def _get_last_learning_rate(self):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            last_lr = self.optimizer.param_groups[0]["lr"]
        else:
            last_lr = self.lr_scheduler.get_last_lr()[0]

        if isinstance(last_lr, torch.Tensor):
            last_lr = last_lr.item()

        return last_lr

    def maybe_log_train_step_metrics(self):
        if self.global_step_last_logged >= self.state.global_step:
            return
        if self.control.should_log:
            dp_size = self.trn_config.data_parallel_size
            tr_loss_div = self.loss / dp_size
            reduced_tr_loss = xm.all_reduce(xm.REDUCE_SUM, tr_loss_div, groups=get_data_parallel_replica_groups())
            reduced_tr_loss = reduced_tr_loss.detach()
            self.loss.zero_()

            xm.mark_step()

            def log_closure(self, reduced_tr_loss, grad_norm):
                # We need to check that self.state.global_step > self._globalstep_last_logged because if two
                # closures are added in a row (which can happen at the end of the training), then it will fail the
                # second time because at this point we will have:
                # self.state.global_step = self._globalstep_last_logged
                if is_logging_process() and self.state.global_step > self.global_step_last_logged:
                    logs: dict[str, float] = {}
                    tr_loss_scalar = reduced_tr_loss.to("cpu").item()

                    logs["loss"] = round(
                        tr_loss_scalar / (self.state.global_step - self.global_step_last_logged), 4
                    )
                    logs["learning_rate"] = self._get_last_learning_rate()

                    if grad_norm is not None:
                        logs["grad_norm"] = (
                            grad_norm.detach().to("cpu").item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        )
                    self.log(logs)

                self.global_step_last_logged = self.state.global_step

            xm.add_step_closure(log_closure, (self, reduced_tr_loss, self.grad_norm))

    def maybe_save_checkpoint(self):
        if self.control.should_save:
            xm.mark_step()

            def save_closure(self, model):
                self._save_checkpoint(model)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            xm.add_step_closure(save_closure, (self,))

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ):

        if resume_from_checkpoint not in [False, None]:
            raise ValueError("`resume_from_checkpoint` is not supported by the NeuronTrainer.")

        args = self.args
        
        # TODO: what's the purpose of this method?
        self.accelerator.free_memory()

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        total_train_batch_size = self.args.train_batch_size * args.gradient_accumulation_steps
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        self.setup_training(train_dataloader, max_steps, num_train_epochs, num_examples, total_train_batch_size)

        for epoch in range(num_train_epochs):
            steps_in_epoch = (
                len_dataloader 
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )

            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            step = -1
            epoch_iterator = iter(train_dataloader)

            num_items_in_batch = 0
            num_items_in_batch_acc = 0

            for step, inputs in enumerate(epoch_iterator):
                xm.mark_step()
                do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                    step + 1
                ) == steps_in_epoch  # Since we perform prefetching, we need to manually set sync_gradients

                for name in inputs:
                    inputs[name] = inputs[name].squeeze(0)

                if self.model_accepts_loss_kwargs and "labels" in inputs:
                    num_items_in_current_batch = inputs["labels"].ne(-100).sum()
                    num_items_in_batch_acc += num_items_in_current_batch

                if not do_sync_step:
                    self.accelerator.gradient_state.sync_gradients = False
                else:
                    self.accelerator.gradient_state.sync_gradients = True

                # TODO: when implementing metrics, save this information.
                main_input_name = getattr(self.model, "main_input_name", "input_ids")
                if main_input_name in inputs:
                    input_tokens = inputs[main_input_name].numel()
                    input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # We accumulate the number of items in the batch over gradient accumulation steps.
                if self.model_accepts_loss_kwargs and "labels" in inputs:
                    if do_sync_step:
                        num_items_in_batch = num_items_in_batch_acc 
                        num_items_in_batch_acc = 0
                    else:
                        num_items_in_batch = 1
                else:
                    num_items_in_batch = None

                loss_step = self.train_step(self.model, inputs)

                # If we can compute the number of items in the batch, we use it to normalize the loss.
                # Otherwise, we divide by gradient_accumulation_steps.
                if num_items_in_batch is None:
                    # Here the loss reduction is "sum" so we need to normalize by the number of items in the batch.
                    loss_step = loss_step / num_items_in_batch
                else:
                    # In this case loss reduction is "mean" so we just need to devide by gradient_accumulation_steps.
                    loss_step = loss_step / self.args.gradient_accumulation_steps

                self.loss += loss_step

                if do_sync_step:
                    xm.mark_step()
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        parameters = (
                            self.model.local_parameters() if isinstance(self.model, NxDPPModel) else self.model.parameters()
                        )
                        self.accelerator.clip_grad_norm_(
                            parameters,
                            args.max_grad_norm,
                            postpone_clipping_to_optimizer_step=True,
                        )

                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()
                    self.grad_norm = self.optimizer.grad_norm

                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()

                    self.optimizer.zero_grad()

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                self.maybe_log_train_step_metrics()
                self.maybe_save_checkpoint()

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    xm.mark_step()
                    break

            # We also need to break out of the nested loop
            if self.control.should_epoch_stop or self.control.should_training_stop:
                xm.mark_step()
                break

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            xm.mark_step()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0 

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        return self.args.process_index == 0

    def log(self, content: dict):
        print("log", content)

