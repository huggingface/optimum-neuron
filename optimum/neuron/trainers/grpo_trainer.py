from typing import Any

import torch
import torch_xla.core.xla_model as xm
from optimum.utils import logging
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ..utils import is_trl_available
from .grpo_config import NeuronGRPOConfig
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION
from neuronx_distributed.pipeline import NxDPPModel


logger = logging.get_logger()


if is_trl_available():
    # Import TRL classes only when available to avoid hard dependency at import time.
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
else:

    class GRPOTrainer:
        """Placeholder used when `trl` is not installed."""


    class GRPOConfig:
        """Placeholder config used when `trl` is not installed."""


# Create a new class that inherits from NeuronTrainer and uses the source methods from GRPOTrainer.
_GRPOTrainer = type(
    "_GRPOTrainer",
    (NeuronTrainer,),
    GRPOTrainer.__dict__.copy()
)


class NeuronGRPOTrainer(_GRPOTrainer):
    """
    GRPOTrainer adapted for Neuron. Only Neuron-specific wiring is kept; GRPO logic is reused from TRL.
    Note: vLLM/offline optimizations should be ported separately in the future.
    """

    def __init__(
        self,
        model: PreTrainedModel | torch.nn.Module | str,
        args: GRPOConfig | None = None,
        data_collator: Any | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        callbacks: list | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, Any] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        **kwargs,
    ):
        if not is_trl_available(required_version=TRL_VERSION): 
            raise RuntimeError(f"Using NeuronGRPOTrainer requires trl=={TRL_VERSION}.")

        args_is_none = args is None
        if args is None:
            args = NeuronGRPOConfig(output_dir="tmp_trainer")
        elif args is not None and args.__class__.__name__ == "NeuronTrainingArguments":
            args_as_dict = args.to_dict()
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = NeuronGRPOConfig(**args_as_dict)

        if args_is_none:
            log_level = args.get_process_log_level() 
            logging.set_verbosity(log_level)
            logging.warning(f"No `GRPOConfig` passed, using `output_dir={args.output_dir}`.")

        NeuronTrainer.__init__(
            self,
            model,
            args, 
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class if tokenizer is None else tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

    def train(self, resume_from_checkpoint: str | bool | None = None):
        return NeuronTrainer.train(self, resume_from_checkpoint=resume_from_checkpoint)

    # must override train_step with grpo loss computation
    def train_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        manager = self.autocast_smart_context_manager()

        if isinstance(model, NxDPPModel):
            with manager:
                loss = model.run_train(**inputs)
            if self.pp_rank != self.pp_size - 1:
                dtype = torch.bfloat16 if self.args.bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
            return loss

        with manager:
            # Delegate GRPO loss computation to TRL's inherited logic
            loss = self.compute_loss(model, inputs, return_outputs=False) 

        if isinstance(num_items_in_batch, torch.Tensor):
            num_items = num_items_in_batch.item()
        else:
            num_items = num_items_in_batch

        if num_items is not None:
            loss = loss / num_items
        else:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss
