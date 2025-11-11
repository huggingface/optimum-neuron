from typing import Any, Iterator
from collections import defaultdict
import inspect
import os
import json
import time
from datetime import datetime
import torch
import torch_xla.core.xla_model as xm
from optimum.utils import logging
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.parallel_layers.parallel_state import get_data_parallel_replica_groups

from ..utils import is_trl_available
from ..utils.misc import is_precompilation
from .grpo_config import NeuronGRPOConfig
from .transformers import NeuronTrainer
from .trl_utils import TRL_VERSION


logger = logging.get_logger()
logger.setLevel(logging.INFO)


if is_trl_available():
    import trl.trainer.utils
    
    def safe_nanmin(tensor: torch.Tensor) -> torch.Tensor:
        """XLA-compatible nanmin that handles empty tensors."""
        if tensor.numel() == 0:
            return torch.tensor(float('nan'), device=tensor.device)
        return torch.min(tensor)
    
    def safe_nanmax(tensor: torch.Tensor) -> torch.Tensor:
        """XLA-compatible nanmax that handles empty tensors."""
        if tensor.numel() == 0:
            return torch.tensor(float('nan'), device=tensor.device)
        return torch.max(tensor)
    
    trl.trainer.utils.nanmin = safe_nanmin
    trl.trainer.utils.nanmax = safe_nanmax
    
    from trl import GRPOConfig, GRPOTrainer
    from trl.data_utils import is_conversational
    from trl.trainer.utils import (
        split_tensor_dict,
        shuffle_sequence_dict,
        selective_log_softmax,
        entropy_from_logits,
        pad,
    )
    _GRPO = GRPOTrainer
else:
    class GRPOTrainer:
        pass
    class GRPOConfig:
        pass
    _GRPO = None


def identity(x):
    return x


def pad_to_neuron_sequence_length(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
    """Pad tensor to be a multiple of 2048 for Neuron flash attention."""
    NEURON_SEQ_LENGTH_MULTIPLE = 2048
    seq_len = tensor.size(1)
    
    if seq_len % NEURON_SEQ_LENGTH_MULTIPLE == 0:
        return tensor
    
    pad_length = NEURON_SEQ_LENGTH_MULTIPLE - (seq_len % NEURON_SEQ_LENGTH_MULTIPLE)
    if tensor.dim() == 2:
        padded = torch.nn.functional.pad(tensor, (0, pad_length), value=pad_value)
    elif tensor.dim() == 3:
        padded = torch.nn.functional.pad(tensor, (0, 0, 0, pad_length), value=pad_value)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
    
    return padded


class NeuronGRPOTrainer(NeuronTrainer):
    """
    GRPO Trainer for Neuron/Trainium devices.
    
    Key adaptations for Neuron:
    1. Generator model loads before distributed training
    2. Tensors kept on CPU for tokenizer operations
    3. XLA-compatible gathering for logging
    4. Sequence padding for Neuron flash attention
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
        reward_funcs = None,
        **kwargs,
    ):
        if not is_trl_available(required_version=TRL_VERSION):
            raise RuntimeError(f"Using NeuronGRPOTrainer requires trl=={TRL_VERSION}.")

        # Extract model name
        if isinstance(model, str):
            self.model_name_or_path = model
        else:
            self.model_name_or_path = getattr(getattr(model, "config", None), "_name_or_path", None)
        
        if not self.model_name_or_path:
            raise ValueError(
                "NeuronGRPOTrainer requires a model ID string. "
                "Pass the model as: model='Qwen/Qwen2-0.5B-Instruct'"
            )

        # Setup args
        args_is_none = args is None
        if args is None:
            args = NeuronGRPOConfig(output_dir="tmp_trainer")
        elif args.__class__.__name__ == "NeuronTrainingArguments":
            args_as_dict = args.to_dict()
            args_as_dict.update({k: getattr(args, k) for k in args_as_dict.keys() if k.endswith("_token")})
            args = NeuronGRPOConfig(**args_as_dict)

        # Set GRPO params early
        self.max_prompt_length = getattr(args, "max_prompt_length", 512)
        self.max_completion_length = getattr(args, "max_completion_length", 128)
        self.num_generations = getattr(args, "num_generations", 1)
        self.temperature = getattr(args, "temperature", 1.0)
        self.top_p = getattr(args, "top_p", 1.0)
        self.top_k = getattr(args, "top_k", None)
        
        if isinstance(model, PreTrainedModel):
            self.model_kwarg_keys = inspect.signature(model.forward).parameters.keys()
        else:
            self.model_kwarg_keys = set()
        
        # Load generator model before distributed setup
        logger.info("[PRE-INIT] Loading generator model before distributed setup...")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self._generator_tokenizer = processing_class or tokenizer or AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        
        # Keep generator on CPU for simplicity and XLA compatibility
        generator_device = getattr(args, "generator_device", "cpu")
        if generator_device == "cuda" and torch.cuda.is_available():
            self.generator_model = self.generator_model.to("cuda")
            logger.info("[PRE-INIT] Generator on CUDA")
        else:
            self.generator_model = self.generator_model.to("cpu")
            logger.info("[PRE-INIT] Generator on CPU")
        
        self.generator_model.eval()
        
        generation_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "pad_token_id": self._generator_tokenizer.pad_token_id,
            "eos_token_id": self._generator_tokenizer.eos_token_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.top_k is not None:
            generation_kwargs["top_k"] = self.top_k
        self.generation_config = GenerationConfig(**generation_kwargs)
        
        logger.info("[PRE-INIT] Generator model ready")
        
        if args_is_none:
            log_level = args.get_process_log_level()
            logging.set_verbosity(log_level)
            logging.warning(f"No `GRPOConfig` passed, using `output_dir={args.output_dir}`.")

        if data_collator is None:
            data_collator = identity

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class if tokenizer is None else tokenizer,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )
        
        # Set model_kwarg_keys after model is prepared
        if hasattr(self, 'model') and self.model is not None:
            model_to_check = self.model
            if isinstance(model_to_check, NxDPPModel):
                model_to_check = model_to_check.module if hasattr(model_to_check, 'module') else model_to_check
            elif hasattr(model_to_check, 'get_base_model'):
                model_to_check = model_to_check.get_base_model()
            
            self.model_kwarg_keys = inspect.signature(model_to_check.forward).parameters.keys()

        # GRPO-specific attributes from config (extract all potential attributes from args)
        self.repetition_penalty = getattr(self.args, "repetition_penalty", None)
        self.min_p = getattr(self.args, "min_p", None)
        self.chat_template_kwargs = getattr(self.args, "chat_template_kwargs", {}) or {}
        self.loss_type = getattr(self.args, "loss_type", "grpo")
        self.mask_truncated_completions = getattr(self.args, "mask_truncated_completions", True)
        self.top_entropy_quantile = getattr(self.args, "top_entropy_quantile", 1.0)
        self.beta = getattr(self.args, "beta", 0.0)
        
        # Epsilon parameters
        self.epsilon_low = getattr(self.args, "epsilon", 0.2)
        _eps_high = getattr(self.args, "epsilon_high", None)
        self.epsilon_high = _eps_high if _eps_high is not None else self.epsilon_low
        
        # Additional GRPO config attributes that TRL might expect (always set with defaults)
        common_grpo_attrs = {
            "alpha": 0.0,
            "gamma": 1.0,
            "cliprange": 0.2,
            "cliprange_value": 0.2,
            "vf_coef": 0.1,
            "ref_free": False,
            "importance_sampling_level": "token",
        }
        for attr, default_value in common_grpo_attrs.items():
            setattr(self, attr, getattr(self.args, attr, default_value))
        
        # Reference model setup (before super().__init__ to avoid distribution issues)

        if self.beta != 0.0 and not self.ref_free:
            logger.info("[PRE-INIT] Loading reference model before distributed setup...")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            try:
                xla_dev = xm.xla_device()
                self.ref_model = self.ref_model.to(xla_dev)
                self.ref_model_on_xla = True
                logger.info(f"[PRE-INIT] Reference model moved to XLA device: {xla_dev}")
            except Exception as e:
                # Fall back gracefully to CPU if XLA device not available
                self.ref_model = self.ref_model.to("cpu")
                self.ref_model_on_xla = False
                logger.info(f"[PRE-INIT] Could not move ref_model to XLA, keeping on CPU: {e}")
            self.ref_model.eval()
            logger.info("[PRE-INIT] Reference model ready")

        else:
            self.ref_model = None
             self.ref_model_on_xla = False
        
        self.mode = "train"
        
        # TRL compatibility attributes
        self.processing_class = self._generator_tokenizer
        self.pad_token_id = self._generator_tokenizer.pad_token_id
        self.eos_token_id = self._generator_tokenizer.eos_token_id
        self.model_wrapped = self.model
        
        # Metrics tracking
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        
        # Generation buffer for batching
        self._buffered_inputs = None
        self._buffer_index = 0
        self._generation_step_counter = 0
        
        # Store reward functions
        self.reward_funcs = reward_funcs or []
        
        # vLLM flags (not used in Neuron but kept for TRL compatibility)
        self.use_vllm = False
        self.vllm_importance_sampling_correction = False
        self.use_transformers_paged = False
        
        # TRL's GRPO trainer uses this for generation frequency
        self.num_iterations = self.args.gradient_accumulation_steps
        
        # JSONL metrics logger - separate file per run with timestamp
        self._jsonl_log_file = None
        self._is_precompilation = is_precompilation()
        if hasattr(self.args, 'output_dir') and self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"grpo_metrics_{timestamp}.jsonl"
            self._jsonl_log_file = os.path.join(self.args.output_dir, filename)
        self._step_start_time = None
        
        if self._is_precompilation:
            logger.warning(
                "⚠️  Running in NEURON_EXTRACT_GRAPHS_ONLY=1 mode (precompilation). "
                "Metrics may be inaccurate as computations are stubbed for graph extraction. "
                "Generation may produce dummy/placeholder outputs. "
                "This mode is for compilation, not actual training. "
                "Remove NEURON_EXTRACT_GRAPHS_ONLY=1 for real training metrics."
            )
        
    def _generate_single_turn(self, prompts, images=None):
        """Override to use CPU generator model for XLA compatibility."""
        generator_device = next(self.generator_model.parameters()).device
        
        # Deduplicate prompts (RepeatSampler duplicates them num_generations times)
        if len(prompts) >= self.num_generations:
            unique_prompts = prompts[::self.num_generations]
            num_unique = len(unique_prompts)
            expected_total = num_unique * self.num_generations
            if expected_total != len(prompts):
                logger.warning(
                    f"Prompt structure mismatch: expected {expected_total}, got {len(prompts)}. "
                    "Generating once per prompt."
                )
                unique_prompts = prompts
                num_unique = len(unique_prompts)
        else:
            unique_prompts = prompts
            num_unique = len(unique_prompts)
        
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": "max_length",
            "padding_side": "left",
            "max_length": self.max_prompt_length,
            "truncation": True,
            "add_special_tokens": False,
        }
        
        # Apply chat template if available (required for Qwen-like models)
        has_chat_template = hasattr(self._generator_tokenizer, 'apply_chat_template')
        
        if has_chat_template:
            try:
                if is_conversational({"prompt": unique_prompts[0]}):
                    generate_inputs = self._generator_tokenizer.apply_chat_template(
                        conversation=unique_prompts,
                        **processor_kwargs,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        **self.chat_template_kwargs,
                    )
                elif hasattr(self._generator_tokenizer, 'chat_template') and self._generator_tokenizer.chat_template:
                    formatted_prompts = []
                    for prompt in unique_prompts:
                        formatted = self._generator_tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        formatted_prompts.append(formatted)
                    generate_inputs = self._generator_tokenizer(text=formatted_prompts, **processor_kwargs)
                else:
                    generate_inputs = self._generator_tokenizer(text=unique_prompts, **processor_kwargs)
            except Exception as e:
                logger.warning(f"Chat template failed, using plain tokenization: {e}")
                generate_inputs = self._generator_tokenizer(text=unique_prompts, **processor_kwargs)
        else:
            generate_inputs = self._generator_tokenizer(text=unique_prompts, **processor_kwargs)
        
        generate_inputs = {
            k: v.to(generator_device) if isinstance(v, torch.Tensor) else v 
            for k, v in generate_inputs.items()
        }
        
        all_prompt_completion_ids = []
        all_prompt_ids = []
        all_prompt_mask = []
        
        with torch.no_grad():
            for prompt_idx in range(num_unique):
                # Extract single prompt inputs
                single_input_ids = generate_inputs["input_ids"][prompt_idx:prompt_idx+1]
                single_attention_mask = generate_inputs["attention_mask"][prompt_idx:prompt_idx+1]
                
                prompt_length = single_input_ids.size(1)
                for gen_idx in range(self.num_generations):
                    prompt_completion_ids = self.generator_model.generate(
                        input_ids=single_input_ids,
                        attention_mask=single_attention_mask,
                        max_new_tokens=self.max_completion_length,
                        pad_token_id=self.pad_token_id,
                        eos_token_id=self.eos_token_id,
                        do_sample=self.generation_config.do_sample,
                        temperature=self.generation_config.temperature,
                        top_p=self.generation_config.top_p,
                        top_k=self.generation_config.top_k,
                    )
                    
                    all_prompt_completion_ids.append(prompt_completion_ids.cpu())
                    all_prompt_ids.append(single_input_ids.cpu())
                    all_prompt_mask.append(single_attention_mask.cpu())
        
        # Stack all generations
        prompt_completion_ids = torch.cat(all_prompt_completion_ids, dim=0)
        prompt_ids = torch.cat(all_prompt_ids, dim=0)
        prompt_mask = torch.cat(all_prompt_mask, dim=0)
        
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        if prompt_completion_ids.size(1) < prompt_length:
            raise ValueError(
                f"prompt_completion_ids length ({prompt_completion_ids.size(1)}) < prompt_length ({prompt_length})"
            )
        is_eos = completion_ids == self.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), completion_ids.size(1), dtype=torch.long)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        sequence_indices = torch.arange(completion_ids.size(1)).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Convert to Python lists
        prompt_ids_list = []
        completion_ids_list = []
        
        for p, pm, c, cm in zip(prompt_ids, prompt_mask, completion_ids, completion_mask):
            valid_prompt_tokens = p[pm.bool()]
            valid_completion_tokens = c[cm.bool()]
            prompt_ids_list.append(valid_prompt_tokens.tolist())
            completion_ids_list.append(valid_completion_tokens.tolist())
        
        return prompt_ids_list, completion_ids_list, None, {}

    # Inherit TRL's _generate method (calls our _generate_single_turn)
    _generate = _GRPO._generate
    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Override to create tensors on CPU for XLA compatibility.
        This avoids the device property issue and ensures tokenizer compatibility.
        """
        # Always use CPU for tensor creation to avoid XLA issues with tokenizer
        device = torch.device("cpu")
        mode = "train" if self.model.training else "eval"
        
        prompts = [x["prompt"] for x in inputs]
        self._last_generated_prompts = prompts
        
        # Generate completions (returns lists)
        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts, None)
        )
        
        # Create tensors on CPU (moved to XLA in get_batch_samples)
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
        
        # Mask truncated completions if needed
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        
        # Concatenate for attention mask
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        
        # Compute old_per_token_logps if needed (policy model logps stay on XLA)
        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self.args.gradient_accumulation_steps % generate_every != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None
        
        # Compute ref_per_token_logps if reference model exists (stays on CPU)
        ref_per_token_logps = None
        entropies_trimmed = None

        if self.beta != 0.0 and self.ref_model is not None:
            with torch.no_grad():
                try:
                    if getattr(self, "ref_model_on_xla", False):
                        xla_dev = xm.xla_device()
                        # pad to Neuron multiple before moving (if needed)
                        pc_ids_xla = pad_to_neuron_sequence_length(prompt_completion_ids.to(xla_dev), self.pad_token_id)
                        attn_mask_xla = pad_to_neuron_sequence_length(attention_mask.to(xla_dev), 0)

                        ref_per_token_logps, ref_entropies = self._get_per_token_logps_and_entropies(
                            self.ref_model,
                            pc_ids_xla,
                            attn_mask_xla,
                            logits_to_keep,
                            batch_size,
                        )

                        # Move XLA tensors to CPU
                        if isinstance(ref_per_token_logps, torch.Tensor) and ref_per_token_logps.device.type == "xla":
                            ref_per_token_logps = ref_per_token_logps.detach().cpu()
                        if ref_entropies is not None and ref_entropies.device.type == "xla":
                            entropies_trimmed = ref_entropies.detach().cpu()

                    else:
                        ref_per_token_logps, entropies_trimmed = self._get_per_token_logps_and_entropies(
                            self.ref_model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                        )

                except Exception as e:
                    logger.warning(f"Reference model logps computation failed: {e}")
                    ref_per_token_logps = None
                    entropies_trimmed = None

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        prompts_for_rewards = [x["prompt"] for x in inputs]
        
        expected_completions = len(prompts_for_rewards)
        if len(completions) != expected_completions:
            logger.warning(
                f"[_generate_and_score_completions] Mismatch: {len(prompts_for_rewards)} prompts but {len(completions)} completions. "
                f"Expected {expected_completions}. This may indicate a data flow issue."
            )
            # Fallback: use decoded prompt_ids (may include chat template)
            prompts_decoded = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
            rewards = self._calculate_rewards(inputs, prompts_decoded, completions, completion_ids_list)
        else:
            rewards = self._calculate_rewards(inputs, prompts_for_rewards, completions, completion_ids_list)
        
        # Validate rewards shape before reshaping
        num_completions = len(completions)
        rewards_size = rewards.numel()
        
        if rewards_size == 0:
            raise ValueError(f"Rewards tensor is empty. Completions: {num_completions}")
        
        if rewards_size % self.num_generations != 0:
            raise ValueError(
                f"Cannot reshape rewards: size {rewards_size} is not divisible by num_generations "
                f"({self.num_generations}). Expected size: num_unique_prompts * {self.num_generations}. "
                f"Completions: {num_completions}"
            )
        
        # Compute advantages (group-relative normalization)
        num_unique_prompts = rewards_size // self.num_generations
        rewards_reshaped = rewards.view(num_unique_prompts, self.num_generations)  # (num_prompts, num_generations)
        mean_grouped_rewards = rewards_reshaped.mean(dim=1)  # (num_prompts,)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)  # (total_samples,)
        advantages = rewards - mean_grouped_rewards
        
        original_prompts_for_debug = prompts[:len(completions)] if len(prompts) >= len(completions) else prompts
        
        # Policy model logps: XLA (preserves computation graph)
        # Ref model logps: CPU (moved to XLA in get_batch_samples when needed)
        output = {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "prompt_mask": prompt_mask,
            "completion_mask": completion_mask,
            "prompt_completion_ids": prompt_completion_ids,
            "attention_mask": attention_mask,
            "logits_to_keep": logits_to_keep,
            "rewards": rewards,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "sampling_per_token_logps": sampling_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
            "_original_prompts": original_prompts_for_debug,
            **extra_fields,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        
        return output
    
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        """Calculate rewards using reward functions."""
        if not self.reward_funcs:
            rewards = torch.zeros(len(completions), device="cpu")
            return rewards
        
        rewards = []
        for reward_func in self.reward_funcs:
            func_rewards = reward_func(prompts, completions)
            if isinstance(func_rewards, list):
                func_rewards = torch.tensor(func_rewards, device="cpu")
            elif not isinstance(func_rewards, torch.Tensor):
                func_rewards = torch.tensor(func_rewards, device="cpu")
            rewards.append(func_rewards)
        
        # Average rewards from multiple functions
        if len(rewards) > 1:
            rewards = torch.stack(rewards).mean(dim=0)
        else:
            rewards = rewards[0]
        
        # Validate rewards shape matches completions
        expected_size = len(completions)
        actual_size = rewards.numel() if rewards.dim() == 0 else len(rewards)
        
        if actual_size != expected_size:
            raise ValueError(
                f"Rewards shape mismatch: expected {expected_size} rewards for {expected_size} completions, "
                f"but got {actual_size}. Rewards shape: {rewards.shape}, Completions: {len(completions)}"
            )
        
        # Ensure 1D tensor
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        elif rewards.dim() > 1:
            rewards = rewards.flatten()
        
        return rewards

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        token_type_ids=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Override to handle CPU->XLA tensor movement and Neuron padding."""
        # Filter out image-related parameters
        image_keys = ["pixel_values", "image_grid_thw", "num_images", "pixel_attention_mask", "image_sizes"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in image_keys}
        
        model_device = next(model.parameters()).device
        
        if model_device.type == "cpu":
            # Ref model: keep tensors on CPU
            input_ids_cpu = input_ids
            attention_mask_cpu = attention_mask
            cpu_kwargs = {}
            if token_type_ids is not None:
                cpu_kwargs["token_type_ids"] = token_type_ids
            
            cpu_kwargs.update(filtered_kwargs)
            
            logps, entropies = _GRPO._get_per_token_logps_and_entropies(
                self,
                model,
                input_ids_cpu,
                attention_mask_cpu,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                **cpu_kwargs,
            )
            return logps, entropies
        
        # Policy model: move tensors to XLA
        xla_device = xm.xla_device()
        
        if input_ids.device.type != "xla":
            input_ids = input_ids.to(xla_device)
        if attention_mask.device.type != "xla":
            attention_mask = attention_mask.to(xla_device)
        
        # Pad to Neuron flash attention requirement (2048 multiple)
        input_ids_padded = pad_to_neuron_sequence_length(input_ids, self.pad_token_id)
        attention_mask_padded = pad_to_neuron_sequence_length(attention_mask, 0)
        
        # Move other tensors to XLA if present
        xla_kwargs = {}
        if token_type_ids is not None:
            xla_kwargs["token_type_ids"] = pad_to_neuron_sequence_length(
                token_type_ids.to(xla_device) if token_type_ids.device.type != "xla" else token_type_ids, 0
            )
        
        xla_kwargs.update(filtered_kwargs)
        
        logps, entropies = _GRPO._get_per_token_logps_and_entropies(
            self,
            model,
            input_ids_padded,
            attention_mask_padded,
            logits_to_keep,
            batch_size=batch_size,
            compute_entropy=compute_entropy,
            **xla_kwargs,
        )
        
        # Keep on XLA to preserve computation graph (CPU->XLA move breaks gradients)
        logps_trimmed = logps[:, :logits_to_keep]
        entropies_trimmed = entropies[:, :logits_to_keep] if entropies is not None else None
        
        return logps_trimmed, entropies_trimmed

    _compute_loss = _GRPO._compute_loss

    def _set_signature_columns_if_needed(self):
        """Override to set GRPO-specific signature columns."""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def get_batch_samples(
        self,
        epoch_iterator: Iterator,
        num_batches: int,
        device: torch.device | None = None,
        prefetch_size: int | None = None,
    ) -> tuple[list[dict[str, Any]], int | torch.Tensor | None]:
        """Override to handle GRPO generation and buffering logic."""
        generate_every = self.args.steps_per_generation * self.num_iterations
        if self._generation_step_counter % generate_every == 0 or self._buffered_inputs is None:
            total_batches_needed = self.args.steps_per_generation
            raw_batches = []
            
            for _ in range(total_batches_needed):
                try:
                    batch = next(epoch_iterator)
                    raw_batches.append(batch)
                except StopIteration:
                    break
            
            if not raw_batches:
                return [], None
            
            raw_samples = []
            for batch in raw_batches:
                if isinstance(batch, list):
                    raw_samples.extend(batch)
                else:
                    raw_samples.append(batch)
            
            generation_batch = self._generate_and_score_completions(raw_samples)
            
            sequence_dict = {}
            scalar_dict = {}
            original_prompts_list = generation_batch.pop("_original_prompts", None)
            
            for key, val in generation_batch.items():
                if val is None:
                    scalar_dict[key] = val
                elif isinstance(val, torch.Tensor) and val.ndim >= 1:
                    sequence_dict[key] = val
                elif isinstance(val, (list, tuple)) and key != "_original_prompts":
                    sequence_dict[key] = val
                else:
                    scalar_dict[key] = val
            
            shuffled_sequences = shuffle_sequence_dict(sequence_dict)
            generation_batches = split_tensor_dict(shuffled_sequences, self.args.steps_per_generation)
            
            if original_prompts_list is not None:
                for batch in generation_batches:
                    batch["_original_prompts"] = original_prompts_list
                    batch.update(scalar_dict)
            else:
                for batch in generation_batches:
                    batch.update(scalar_dict)
            
            self._buffered_inputs = generation_batches
            self._buffer_index = 0
        
        current_batch = self._buffered_inputs[self._buffer_index]
        self._buffer_index += 1
        self._generation_step_counter += 1
        
        if self._buffer_index >= len(self._buffered_inputs):
            self._buffer_index = 0
        
        # Move tensors to XLA device for training
        if device is not None and device.type == "xla":
            current_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in current_batch.items()
            }
        
        return [current_batch], None

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | torch.Tensor | None = None,
    ):
        """Compute GRPO loss."""
        if return_outputs:
            raise ValueError("return_outputs=True is not supported for GRPO")
        
        loss = self._compute_loss(model, inputs) 
        return loss

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Training step that works with GRPO's prepared inputs."""
        self._step_start_time = time.time()
        manager = self.autocast_smart_context_manager()

        # Extract metrics from inputs before computation
        rewards = inputs.get("rewards")
        advantages = inputs.get("advantages")
        
        with manager:
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if isinstance(model, NxDPPModel):
            self.accelerator.backward(loss)
            if self.pp_rank != self.pp_size - 1:
                dtype = torch.bfloat16 if self.args.bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
        else:
            self.accelerator.backward(loss)
        
        # Compute and log metrics
        mode = "train" if model.training else "eval"
        
        # Log loss
        if loss is not None and isinstance(loss, torch.Tensor):
            loss_item = loss.item() if loss.numel() == 1 else loss.mean().item()
            self._metrics[mode]["loss"].append(loss_item)
        
        # Log rewards statistics
        if rewards is not None and isinstance(rewards, torch.Tensor):
            if rewards.numel() > 0:
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    rewards_gathered = self.accelerator.gather(rewards)
                    rewards_for_stats = rewards_gathered.cpu()
                else:
                    rewards_for_stats = rewards.cpu() if rewards.device.type == "xla" else rewards
                
                if rewards_for_stats.numel() > 0:
                    rewards_flat = rewards_for_stats.flatten()
                    self._metrics[mode]["rewards/mean"].append(rewards_flat.mean().item())
                    self._metrics[mode]["rewards/max"].append(rewards_flat.max().item())
                    self._metrics[mode]["rewards/min"].append(rewards_flat.min().item())
                    self._metrics[mode]["rewards/std"].append(rewards_flat.std().item())
        
        # Log advantages statistics
        if advantages is not None and isinstance(advantages, torch.Tensor):
            if advantages.numel() > 0:
                if hasattr(self, 'accelerator') and self.accelerator is not None:
                    advantages_gathered = self.accelerator.gather(advantages)
                    advantages_for_stats = advantages_gathered.cpu()
                else:
                    advantages_for_stats = advantages.cpu() if advantages.device.type == "xla" else advantages
                
                if advantages_for_stats.numel() > 0:
                    advantages_flat = advantages_for_stats.flatten()
                    self._metrics[mode]["advantages/mean"].append(advantages_flat.mean().item())
                    self._metrics[mode]["advantages/max"].append(advantages_flat.max().item())
                    self._metrics[mode]["advantages/min"].append(advantages_flat.min().item())
        
        # Compute gradient norm
        grad_norm = None
        try:
            total_norm = 0.0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_data = p.grad.data
                    try:
                        if hasattr(grad_data, 'cpu'):
                            param_norm = grad_data.cpu().norm(2).item()
                        else:
                            param_norm = grad_data.norm(2).item()
                        total_norm += param_norm ** 2
                        param_count += 1
                    except Exception:
                        pass
            if param_count > 0:
                grad_norm = total_norm ** (1. / 2)
                self._metrics[mode]["grad_norm"].append(grad_norm)
        except Exception as e:
            logger.debug(f"Could not compute grad norm: {e}")
        
        loss_item = None
        if loss is not None and isinstance(loss, torch.Tensor):
            try:
                if hasattr(loss, 'cpu'):
                    loss_cpu = loss.cpu()
                    loss_item = loss_cpu.item() if loss_cpu.numel() == 1 else loss_cpu.mean().item()
                else:
                    loss_item = loss.item() if loss.numel() == 1 else loss.mean().item()
            except Exception:
                try:
                    loss_item = loss.item()
                except Exception:
                    loss_item = None
        
        reward_mean = None
        reward_std = None
        if rewards is not None and isinstance(rewards, torch.Tensor) and rewards.numel() > 0:
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                rewards_gathered = self.accelerator.gather(rewards)
                rewards_for_stats = rewards_gathered.cpu()
            else:
                rewards_for_stats = rewards.cpu() if rewards.device.type == "xla" else rewards
            if rewards_for_stats.numel() > 0:
                rewards_flat = rewards_for_stats.flatten()
                reward_mean = rewards_flat.mean().item()
                reward_std = rewards_flat.std().item()
        
        kl_loss = None
        if mode in self._metrics and "kl" in self._metrics[mode] and len(self._metrics[mode]["kl"]) > 0:
            kl_loss = self._metrics[mode]["kl"][-1]
        
        learning_rate = None
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            try:
                learning_rate = self.lr_scheduler.get_last_lr()[0]
            except Exception:
                pass
        
        tokens_generated = None
        if "completion_ids" in inputs:
            completion_ids = inputs["completion_ids"]
            if isinstance(completion_ids, torch.Tensor):
                tokens_generated = completion_ids.numel()
            elif isinstance(completion_ids, list):
                tokens_generated = sum(len(ids) if isinstance(ids, (list, torch.Tensor)) else 0 for ids in completion_ids)
        
        time_per_step_s = None
        if self._step_start_time is not None:
            time_per_step_s = time.time() - self._step_start_time
        
        # Log to JSONL
        if self._jsonl_log_file and hasattr(self, 'state') and self.state is not None:
            step = self.state.global_step
            epoch = getattr(self.state, 'epoch', 0.0)
            
            metrics = {
                "timestamp_iso": datetime.now().isoformat(),
                "step": step,
                "epoch": epoch,
                "grpo_loss": loss_item,
                "policy_loss": loss_item,
                "kl_loss": kl_loss,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "grad_norm": grad_norm,
                "learning_rate": learning_rate,
                "time_per_step_s": time_per_step_s,
                "tokens_generated": tokens_generated,
                "is_precompilation": self._is_precompilation,
            }
            
            if hasattr(self, 'accelerator') and self.accelerator is not None:
                if self.accelerator.is_main_process:
                    with open(self._jsonl_log_file, 'a') as f:
                        f.write(json.dumps(metrics) + '\n')
            else:
                with open(self._jsonl_log_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
        
        return loss

    def train(self, resume_from_checkpoint: str | bool | None = None):
        """Use NeuronTrainer's training loop."""
        return super().train(resume_from_checkpoint=resume_from_checkpoint)