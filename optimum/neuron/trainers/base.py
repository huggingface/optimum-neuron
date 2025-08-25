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
import inspect
import math
import os
import shutil
import sys
import time
import warnings
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
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
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import is_datasets_available
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.tokenizatgion_utils_base import PreTrainedTokenizerBase
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
from transformers.integrations import hp_params. get_reporting_integration_callbacks
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
    LabelSmoother,
    find_batch_size,
    get_dataloader_sampler,
    nested_concat,
    nested_numpify,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
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
    is_main_worker_for_metrics,
    is_main_worker_for_metrics_method,
    patch_generation_mixin_to_neuron_generation_mixin,
    skip_first_batches,
)
from .training_args import NeuronTrainingArguments


logger = logging.get_logger("transformers.trainer")

if is_datasets_available():
    import datasets

TRL_VERSION = "0.11.4"

KEEP_HF_HUB_PROGRESS_BARS = os.environ.get("KEEP_HF_HUB_PROGRESS_BARS")
if KEEP_HF_HUB_PROGRESS_BARS is None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

transformers_get_optimizer_cls_and_kwargs = Trainer.get_optimizer_cls_and_kwargs

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback


class NeuronBaseTrainer:
    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | "datasets.Dataset" | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | "datasets.Dataset" | None = None,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):

        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = NeuronTrainingArguments(output_dir=output_dir)

        self.args = args
        self.compute_loss_func = compute_loss_func

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

        self.model = model

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

class _TrainerForNeuron:
    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: TrainingArguments = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | "datasets.Dataset" | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | "datasets.Dataset" | None = None,
        processing_class: PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = NeuronTrainingArguments(output_dir=output_dir)
        if args.batch_eval_metrics and compute_metrics is not None:
            if "compute_result" not in inspect.signature(compute_metrics).parameters.keys():
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        if args.eval_strategy is not None and args.eval_strategy != "no" and eval_dataset is None:
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy} but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. "
            )
        if args.save_strategy == SaveStrategy.BEST or args.load_best_model_at_end:
            if args.metric_for_best_model is None:
                raise ValueError(
                    "`args.metric_for_best_model` must be provided when using 'best' save_strategy or if `args.load_best_model_at_end` is set to `True`."
                )

        self.args = args
        self.compute_loss_func = compute_loss_func
        # Seed must be set before instantiating the model when using model
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        self.hp_name = None
        self.deepspeed = None
        self.is_in_train = False

        self.create_accelerator_and_postprocess()

        # memory metrics - must set up as early as possible
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        # set the correct log level depending on the node
        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)

        # force device and distributed setup init explicitly
        args._setup_devices

        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                warnings.warn(
                    "`Trainer` requires either a `model` or `model_init` argument, but not both. `model_init` will"
                    " overwrite your model when calling the `train` method. This will become a fatal error in the next"
                    " release.",
                    FutureWarning,
                )
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        if getattr(model, "is_parallelizable", False) and getattr(model, "model_parallel", False):
            self.is_model_parallel = True
        else:
            self.is_model_parallel = False

        if getattr(model, "hf_device_map", None) is not None:
            devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])
            else:
                self.is_model_parallel = False

            # warn users
            if self.is_model_parallel:
                logger.info(
                    "You have loaded a model on multiple GPUs. `is_model_parallel` attribute will be force-set"
                    " to `True` to avoid any unexpected behavior such as device placement mismatching."
                )

        if self.args.use_liger_kernel:
            if is_liger_kernel_available():
                from liger_kernel.transformers import _apply_liger_kernel_to_instance

                if isinstance(model, PreTrainedModel):
                    # Patch the model with liger kernels. Use the default kernel configurations.
                    _apply_liger_kernel_to_instance(model=model)
                elif hasattr(model, "get_base_model") and isinstance(model.get_base_model(), PreTrainedModel):
                    # Patch the base model with liger kernels where model is a PeftModel. Use the default kernel configurations.
                    _apply_liger_kernel_to_instance(model=model.get_base_model())
                else:
                    logger.warning(
                        "The model is not an instance of PreTrainedModel. No liger kernels will be applied."
                    )
            else:
                raise ImportError(
                    "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                    "Please install it with `pip install liger-kernel`"
                )

        _is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
            model, "_hf_peft_config_loaded", False
        )
        _quantization_method_supports_training = (
            getattr(model, "hf_quantizer", None) is not None and model.hf_quantizer.is_trainable
        )

        _is_model_quantized_and_qat_trainable = getattr(model, "hf_quantizer", None) is not None and getattr(
            model.hf_quantizer, "is_qat_trainable", False
        )

        # Filter out quantized + compiled models
        if _is_quantized_and_base_model and hasattr(model, "_orig_mod"):
            raise ValueError(
                "You cannot fine-tune quantized model with `torch.compile()` make sure to pass a non-compiled model when fine-tuning a quantized model with PEFT"
            )

        # At this stage the model is already loaded
        if _is_quantized_and_base_model and not _is_peft_model(model) and not _is_model_quantized_and_qat_trainable:
            raise ValueError(
                "You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of"
                " the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft"
                " for more details"
            )
        elif _is_quantized_and_base_model and not _quantization_method_supports_training:
            raise ValueError(
                f"The model you are trying to fine-tune is quantized with {model.hf_quantizer.quantization_config.quant_method}"
                " but that quantization method do not support training. Please open an issue on GitHub: https://github.com/huggingface/transformers"
                f" to request the support for training support for {model.hf_quantizer.quantization_config.quant_method}"
            )

        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]
        if len(args.fsdp) > 0:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if not args.fsdp_config["xla"] and args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise ValueError("Using fsdp only works in distributed training.")

        # one place to sort out whether to place the model on device or not
        # postpone switching model to cuda when:
        # 1. MP - since we are trying to fit a much bigger than 1 gpu model
        # 2. fp16-enabled DeepSpeed loads the model in half the size and it doesn't need .to() anyway,
        #    and we only use deepspeed for training at the moment
        # 3. full bf16 or fp16 eval - since the model needs to be cast to the right dtype first
        # 4. FSDP - same as MP
        self.place_model_on_device = args.place_model_on_device
        if (
            self.is_model_parallel
            or self.is_deepspeed_enabled
            or ((args.fp16_full_eval or args.bf16_full_eval) and not args.do_train)
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
        ):
            self.place_model_on_device = False

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

        # Bnb Quantized models doesn't support `.to` operation.
        if (
            self.place_model_on_device
            and not getattr(model, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES
        ):
            self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # later use `self.model is self.model_wrapped` to check if it's wrapped or not
        self.model_wrapped = model
        self.model = model

        # Just in case the model was wrapped outside of the `Trainer`
        unwrapped_model = self.accelerator.unwrap_model(model)
        model_forward = (
            unwrapped_model.forward
            if not _is_peft_model(unwrapped_model)
            else unwrapped_model.get_base_model().forward
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

        self.neftune_noise_alpha = args.neftune_noise_alpha

        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs
        if self.optimizer_cls_and_kwargs is not None and self.optimizer is not None:
            raise RuntimeError("Passing both `optimizers` and `optimizer_cls_and_kwargs` arguments is incompatible.")
        if model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_xla_available() and self.optimizer is not None:
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
        if (self.is_fsdp_xla_enabled or self.is_fsdp_enabled) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # Will be set to True by `self._setup_loggers()` on first call to `self.log()`.
        self._loggers_initialized = False

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

        if (
            train_dataset is not None
            and isinstance(train_dataset, torch.utils.data.IterableDataset)
            and args.group_by_length
        ):
            raise ValueError("the `--group_by_length` option is only available for `Dataset`, not `IterableDataset")

        self._signature_columns = None

        # Mixed precision setup
        self.use_apex = False
        self.use_cpu_amp = False

        # Mixed precision setup for SageMaker Model Parallel
        if is_sagemaker_mp_enabled():
            # BF16 + model parallelism in SageMaker: currently not supported, raise an error
            if args.bf16:
                raise ValueError("SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead ")

            if IS_SAGEMAKER_MP_POST_1_10:
                # When there's mismatch between SMP config and trainer argument, use SMP config as truth
                if args.fp16 != smp.state.cfg.fp16:
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        f"but FP16 provided in trainer argument is {args.fp16}, "
                        f"setting to {smp.state.cfg.fp16}"
                    )
                    args.fp16 = smp.state.cfg.fp16
            else:
                # smp < 1.10 does not support fp16 in trainer.
                if hasattr(smp.state.cfg, "fp16"):
                    logger.warning(
                        f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                        "but SageMaker Model Parallelism < 1.10 does not support FP16 in trainer."
                    )
        if (args.fp16 or args.bf16) and args.half_precision_backend == "auto":
            if args.device == torch.device("cpu"):
                if args.fp16:
                    if not is_torch_greater_or_equal_than_2_3:
                        raise ValueError("Tried to use `fp16` but it is not supported on cpu")
                else:
                    args.half_precision_backend = "cpu_amp"
            logger.info(f"Using {args.half_precision_backend} half precision backend")

        if (args.fp16 or args.bf16) and not (self.is_deepspeed_enabled or is_sagemaker_mp_enabled()):
            # deepspeed and SageMaker Model Parallel manage their own half precision
            if args.half_precision_backend == "cpu_amp":
                self.use_cpu_amp = True
                self.amp_dtype = torch.bfloat16
            elif args.half_precision_backend == "apex":
                if not is_apex_available():
                    raise ImportError(
                        "Using FP16 with APEX but APEX is not installed, please refer to"
                        " https://www.github.com/nvidia/apex."
                    )
                self.use_apex = True

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
        # Internal variable to count flos in each process, will be accumulated in `self.state.total_flos` then
        # returned to 0 every time flos need to be logged
        self.current_flos = 0
        self.hp_search_backend = None
        if _is_peft_model(self.model) and self.args.label_names is None:
            logger.warning(
                f"No label_names provided for model class `{self.model.__class__.__name__}`."
                " Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`."
                " Note that empty label_names list will be used instead."
            )
        default_label_names = find_labels(self.model.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(self.model.__class__)
        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # Internal variables to help with automatic batch size reduction
        self._train_batch_size = args.train_batch_size
        self._created_lr_scheduler = False

        # very last
        self._memory_tracker.stop_and_update_metrics()

        # torch.compile
        if args.torch_compile and not is_torch_compile_available():
            raise RuntimeError("Using torch.compile requires PyTorch 2.0 or higher.")

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)
        if self.is_fsdp_xla_v2_enabled:
            if not IS_XLA_FSDPV2_POST_2_2:
                raise ValueError("FSDPv2 requires `torch_xla` 2.2 or higher.")
            # Prepare the SPMD mesh that is going to be used by the data loader and the FSDPv2 wrapper.
            # Tensor axis is just a placeholder where it will not be used in FSDPv2.
            num_devices = xr.global_runtime_device_count()
            xs.set_global_mesh(xs.Mesh(np.array(range(num_devices)), (num_devices, 1), axis_names=("fsdp", "tensor")))
        self.is_fsdp_xla_v1_enabled = self.is_fsdp_xla_enabled and not self.is_fsdp_xla_v2_enabled


        # LIMIT
        if not isinstance(self, Trainer):
            raise TypeError(f"{self.__class__.__name__} can only be mixed with Trainer subclasses.")

        training_args = kwargs.get("args", None)
        if training_args is None and len(args) >= 2:
            training_args = args[1]

        self.use_amp = False
        if training_args is not None:
            if training_args.bf16:
                if training_args.half_precision_backend == "amp":
                    self.use_amp = True

        # It is a mixin-in, we call the parent constructor, which is expected to be a children of Trainer.
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, NeuronTrainingArguments):
            raise ValueError(
                f"The NeuronTrainer only accept NeuronTrainingArguments, but {type(self.args)} was provided."
            )

        # We need to change which process can be seen as "world process zero" to make sure the proper metrics
        # (eg.g loss) are logged and sent to the callbacks (for instance WandbCallback).
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=is_main_worker_for_metrics(),
        )

        if self.args.local_rank <= 0:
            logger.setLevel(logging.INFO)

        # Make the model Neuron-compatible for generation.
        patch_generation_mixin_to_neuron_generation_mixin(self.model)

    @property
    def mp_enabled(self):
        return self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM

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

        dataloader_config = DataLoaderConfiguration(
            split_batches=accelerator_config.pop("split_batches"),
            dispatch_batches=accelerator_config.pop("dispatch_batches"),
            even_batches=accelerator_config.pop("even_batches"),
            use_seedable_sampler=accelerator_config.pop("use_seedable_sampler"),
        )

        non_blocking = accelerator_config.pop("non_blocking")
        if non_blocking and not self.args.dataloader_pin_memory:
            logger.warning(
                "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
            )
        dataloader_config.non_blocking = non_blocking

        args = {
            "deepspeed_plugin": None, # We don't use deepspeed plugin in NeuronTrainer
            "dataloader_config": dataloader_config,
        }

        # create accelerator object
        self.accelerator = NeuronAccelerator(
            *args,
            trn_config=self.args.trn_config,
            zero_1=self.args.zero_1,
            mixed_precision="bf16" if self.args.bf16 else "no",
            autocast_backend=self.args.half_precision_backend,
        )

        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

    @requires_torch_neuronx
    def synchronize_hub_cache(self):
        cache_path = get_neuron_cache_path()
        repo_id = get_hf_hub_cache_repos()[0]
        if not self.args.skip_cache_push:
            try:
                synchronize_hub_cache(cache_path=cache_path, cache_repo_id=repo_id)
            except Exception as e:
                logger.warning(
                    f"Failed to synchronize the hub cache for {repo_id}. This is not a critical error, but it prevents"
                    f"compilation caching with the Hugging Face Hub. Error: {e}"
                )

    def _get_train_sampler(self) -> torch.utils.data.Sampler | None:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.mp_enabled:
            if self.args.group_by_length:
                raise ValueError("LengthGroupedSampler is currently not supported with model parallelism.")

            return torch.utils.data.RandomSampler(self.train_dataset)

        return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> torch.utils.data.Sampler | None:
        return torch.utils.data.SequentialSampler(eval_dataset)

    def get_num_trainable_parameters(self):
        return get_model_param_count(self.model, trainable_only=True)

    def create_optimizer(self):
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

            # ** Difference with the original `create_optimizer` method **
            # We removed the part handling bitsandbyte optimizers, as it is not supported in Neuron.

        return self.optimizer

    def _prepare_input(self, data: torch.Tensor | Any) -> torch.Tensor | Any:
        # When pipeline parallelism is enabled, we should not put any tensor on device.
        # It is handled by the NxDPPModel class.
        if self.args.trn_config.pipeline_parallel_size > 1:
            return data
        return super()._prepare_input(data)

    def _prepare_inputs(self, inputs: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        inputs = super()._prepare_inputs(inputs)
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        from neuronx_distributed.pipeline import NxDPPModel

        if isinstance(model, NxDPPModel):
            inputs = self._prepare_inputs(inputs)
            loss = model.run_train(**inputs)
        else:
            loss = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        return loss

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

    def training_step(
        self, model: torch.nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch=None
    ) -> torch.Tensor:
        from neuronx_distributed.pipeline import NxDPPModel

        if isinstance(model, NxDPPModel):
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_pipeline_model_parallel_rank,
                get_pipeline_model_parallel_size,
            )

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if get_pipeline_model_parallel_rank() != get_pipeline_model_parallel_size() - 1:
                use_bf16 = self.accelerator.state.mixed_precision == "bf16"
                dtype = torch.bfloat16 if use_bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())

            if num_items_in_batch is None:
                loss = loss / self.args.gradient_accumulation_steps
        else:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        return loss

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        from neuronx_distributed.pipeline import NxDPPModel

        if isinstance(model, NxDPPModel):
            if not prediction_loss_only:
                raise ValueError("Only the prediction loss can be returned when doing pipeline parallelism.")
            loss = model.run_eval(**inputs)
            if loss is None:
                use_bf16 = self.accelerator.state.mixed_precision == "bf16"
                dtype = torch.bfloat16 if use_bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
            return (loss, None, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        # We always reduce the loss, even when we do not use it to avoid a new graph.
        # This communication is not costly.

        if self.state.global_step > self._globalstep_last_logged:
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_data_parallel_replica_groups,
                get_data_parallel_size,
                model_parallel_is_initialized,
            )

            if model_parallel_is_initialized():
                dp_size = get_data_parallel_size()
            else:
                dp_size = xr.world_size()

            tr_loss_div = tr_loss / dp_size

            reduced_tr_loss = xm.all_reduce(xm.REDUCE_SUM, tr_loss_div, groups=get_data_parallel_replica_groups())

            reduced_tr_loss = reduced_tr_loss.detach()

            if self.control.should_log:
                xm.mark_step()
                tr_loss.zero_()

                def log_closure(self, reduced_tr_loss, grad_norm):
                    # We need to check that self.state.global_step > self._globalstep_last_logged because if two
                    # closures are added in a row (which can happen at the end of the training), then it will fail the
                    # second time because at this point we will have:
                    # self.state.global_step = self._globalstep_last_logged
                    if is_main_worker_for_metrics() and self.state.global_step > self._globalstep_last_logged:
                        logs: dict[str, float] = {}
                        tr_loss_scalar = reduced_tr_loss.to("cpu").item()

                        logs["loss"] = round(
                            tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4
                        )
                        logs["learning_rate"] = self._get_learning_rate()

                        if grad_norm is not None:
                            logs["grad_norm"] = (
                                grad_norm.detach().to("cpu").item()
                                if isinstance(grad_norm, torch.Tensor)
                                else grad_norm
                            )

                        self._total_loss_scalar += tr_loss_scalar
                        self.store_flos()
                        self.log(logs, start_time)

                    self._globalstep_last_logged = self.state.global_step

                xm.add_step_closure(log_closure, (self, reduced_tr_loss, grad_norm))

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            xm.mark_step()

            def save_closure(self, model, trial):
                self._save_checkpoint(model, trial)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            xm.add_step_closure(save_closure, (self, model, trial))

    def _save_xla(self, output_dir: str | None = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if is_main_worker():
            logger.info(f"Saving model checkpoint to {output_dir}")

            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        xm.rendezvous("saving_checkpoint")
        if self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            if is_main_worker():
                logger.info(
                    "Model parallelism is enabled, saving the model sharded state dict instead of the full state dict."
                )

            model_to_save = self.model.original_torch_module if isinstance(self.model, NxDPPModel) else self.model
            model_to_save.save_pretrained(
                output_dir,
                optimizer=self.optimizer if not self.args.save_only_model else None,
            )
        else:
            if not isinstance(self.model, PreTrainedModel):
                if isinstance(unwrap_model(self.model), PreTrainedModel):
                    unwrap_model(self.model).save_pretrained(
                        output_dir,
                        is_main_process=self.args.should_save,
                        state_dict=self.model.state_dict(),
                        save_function=xm.save,
                    )
                else:
                    if is_main_worker():
                        logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                    state_dict = self.model.state_dict()
                    xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
            else:
                self.model.save_pretrained(
                    output_dir,
                    is_main_process=self.args.should_save,
                    save_function=xm.save,
                )

        if self.processing_class is not None and self.args.should_save:
            self.processing_class.save_pretrained(output_dir)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if not is_precompilation():  # Avoid unnecessary model saving during precompilation
            with patch_neuron_cc_wrapper():
                with hub_neuronx_cache(cache_dir=get_neuron_cache_path()):
                    if output_dir is None:
                        output_dir = self.args.output_dir

                    self._save_xla(output_dir)

            # Push to the Hub when `save_model` is called by the user.
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        elif is_main_worker():
            logger.info("Skipping trainer.save_model() while running under neuron_parallel_compile")

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)

        self.save_model(output_dir, _internal_call=True)

        # The optimizer state is saved in the shard alongside with the model parameters when doing model-parallelism.
        if self.accelerator.distributed_type is not NeuronDistributedType.MODEL_PARALLELISM:
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)

        with warnings.catch_warnings(record=True) as caught_warnings:
            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

        if not self.args.save_only_model:
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # It has been handled during model parallelization.
        if self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return
        if self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            lr_scheduler_state = torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu")
            xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)
            self.lr_scheduler.load_state_dict(lr_scheduler_state)
            optimizer_state = torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location="cpu")
            xm.send_cpu_data_to_device(optimizer_state, self.args.device)
            self.optimizer.load_state_dict(optimizer_state)
        else:
            return super()._load_optimizer_and_scheduler(checkpoint)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        from neuronx_distributed.pipeline import NxDPPModel

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if is_main_worker():
            logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.dp_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
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

        model = self._wrap_model(self.model_wrapped)
        model = self.accelerator.prepare(model)
        self.model = model

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        if not isinstance(model, NxDPPModel):
            self.model.train()

        if hasattr(self.lr_scheduler, "step"):
            self.optimizer = self.accelerator.prepare(self.optimizer)
        else:
            # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
            self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.optimizer, self.lr_scheduler)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # Train!
        parameter_count = get_model_param_count(model, trainable_only=True)
        if is_main_worker():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples:,}")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            if self.args.per_device_train_batch_size != self._train_batch_size:
                logger.info(
                    f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
                )
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
            )
            logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Number of trainable parameters = {parameter_count:,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            if is_main_worker():
                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        # We need to change which process can be seen as "world process zero" to make sure the proper metrics
        # (eg.g loss) are logged and sent to the callbacks (for instance WandbCallback).
        self.state.is_world_process_zero = is_main_worker_for_metrics()

        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        self.optimizer.zero_grad()
        grad_norm: float | None = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Mark step before training to materialize any tensor before creating the training graph.
        xm.mark_step()

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [torch.utils.data.RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    from accelerate.data_loader import SeedableRandomSampler

                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    xm.mark_step()
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (
                        step + 1
                    ) == steps_in_epoch  # Since we perform prefetching, we need to manually set sync_gradients
                    if not do_sync_step:
                        self.accelerator.gradient_state.sync_gradients = False
                    else:
                        self.accelerator.gradient_state.sync_gradients = True

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )

                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i == len(batch_samples) - 1
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    if args.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is "
                                f"{tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state.sync_gradients = True
                        xm.mark_step()

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            parameters = (
                                model.local_parameters() if isinstance(model, NxDPPModel) else model.parameters()
                            )
                            self.accelerator.clip_grad_norm_(
                                parameters,
                                args.max_grad_norm,
                                postpone_clipping_to_optimizer_step=True,
                            )

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()
                        grad_norm = self.optimizer.grad_norm

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        self.optimizer.zero_grad()

                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

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
                if is_main_worker():
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            xm.mark_step()
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        if is_main_worker():
            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            xm.rendezvous("load_best_model")

            self._load_best_model()

        # add remaining tr_loss
        loss_scalar = tr_loss.to("cpu").item()
        self._total_loss_scalar += loss_scalar
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        if is_main_worker_for_metrics():
            self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    if is_main_worker():
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        from neuronx_distributed.parallel_layers.parallel_state import get_data_parallel_size
        from neuronx_distributed.pipeline import NxDPPModel

        # This will prepare the model if it was not prepared before.
        # This is needed for example for TP when we performing only evaluation (no training):
        #   1. The model needs to be loaded if it was lazy loaded.
        #   2. The model needs to be parallelized.
        model = self.accelerator.prepare_model(self.model)

        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        is_nxdppmodel = isinstance(model, NxDPPModel)
        if not is_nxdppmodel:
            model = self._wrap_model(model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train and not is_nxdppmodel:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        if is_main_worker():
            logger.info(f"***** Running {description} *****")
            dp_size = get_data_parallel_size()
            logger.info(f"  Num data parallel workers = {dp_size}")
            if has_length(dataloader):
                num_examples = self.num_examples(dataloader)
                total_num_examples = num_examples * dp_size
                logger.info(f"  Per data parallel worker num examples = {num_examples}")
                logger.info(f"  Total num examples = {total_num_examples}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

        if not is_nxdppmodel:
            model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            if is_nxdppmodel and observed_batch_size % model.num_microbatches != 0:
                if is_main_worker() == 0:
                    logger.warning(
                        "Skipping the evaluation step because the pipeline number of microbatches "
                        f"({model.num_microbatches}) does not divide the batch size ({observed_batch_size})."
                    )
                continue

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial=None,  # No type-annotation for this one because it is related to the optuna package.
        ignore_keys_for_eval: list[str] | None = None,
        **kwargs,
    ):
        with hub_neuronx_cache(cache_dir=get_neuron_cache_path()):
            result = super().train(
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                **kwargs,
            )
        if not is_precompilation():
            self.synchronize_hub_cache()
        return result

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        with hub_neuronx_cache(cache_dir=get_neuron_cache_path()):
            with self.args.world_size_as_dp_size():
                result = super().evaluate(
                    eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
                )
        if not is_precompilation():
            self.synchronize_hub_cache()
        return result

    def predict(
        self, test_dataset: Dataset, ignore_keys: list[str] | None = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        with hub_neuronx_cache(cache_dir=get_neuron_cache_path()):
            with self.args.world_size_as_dp_size():
                result = super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        if not is_precompilation():
            self.synchronize_hub_cache()
        return result

    @patch_within_function(("transformers.Trainer.is_world_process_zero", is_main_worker_for_metrics_method))
    def save_metrics(self, split, metrics, combined=True):
        return super().save_metrics(split, metrics, combined=combined)

    @patch_within_function(("transformers.Trainer.is_world_process_zero", is_main_worker_for_metrics_method))
    def save_state(self):
        return super().save_state()

    def push_to_hub(
        self,
        commit_message: str | None = "End of training",
        blocking: bool = True,
        token: str | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> str:
        if is_precompilation():
            logger.info("Push to hub skipped during precompilation.")
            return ""
        else:
            return super().push_to_hub(commit_message, blocking, token, revision, **kwargs)
