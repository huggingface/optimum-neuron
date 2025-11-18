from typing import Any, Iterator

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

_GRPO = _GRPOTrainer # TODO 

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

    def train(self, resume_from_checkpoint: str | bool | None = None):
        """Use NeuronTrainer's training loop."""
        return super().train(resume_from_checkpoint=resume_from_checkpoint)