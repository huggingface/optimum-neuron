import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from modules.autobucketing import slice_lhs, slice_rhs  # noqa: E402
from modules.checkpoint import load_state_dict
from modules.config import NeuronInferenceConfig
from modules.gqa import (  # noqa: E402
    determine_sharding_strategy,  # noqa: E402
    get_shardable_head_counts,  # noqa: E402
)  # noqa: E402
from modules.model_wrapper import (  # noqa: E402
    CONTEXT_ENCODING_MODEL_TAG,  # noqa: E402
    TOKEN_GENERATION_MODEL_TAG,  # noqa: E402
    ContextEncodingModelExporter,
    ModelWrapper,  # noqa: E402
    TokenGenerationModelExporter,
)
from modules.sampling import Sampler  # noqa: E402
from neuronx_distributed.parallel_layers import parallel_state, utils  # noqa: E402
from neuronx_distributed.trace.model_builder import ModelBuilder
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import (
    GenerationConfig,
    GenerationMixin,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput


SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]


class NeuronBaseModel(PreTrainedModel):
    """
    Base model that NeuronXXXModel classes inherit from.

    The forward() function will be traced and compiled by NxD.
    """

    SEQ_DIM = 2

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.batch_size = config.batch_size
        self.n_positions = config.n_positions
        self.vocab_size = config.vocab_size
        self.padding_side = config.padding_side
        self.max_length = config.max_length

        self.setup_attr_for_model(config)
        self.init_model(config)
        self.init_inference_optimization(config)
        self.post_init()

    def setup_attr_for_model(self, config: PretrainedConfig):
        """
        Please provide model-specific definition for the following attributes
            self.on_device_sampling
            self.tp_degree
            self.hidden_size
            self.num_attention_heads
            self.num_key_value_heads
            self.max_batch_size
        """
        raise NotImplementedError("setup_attr_for_model() is not implemented")

    def init_model(self, config: PretrainedConfig):
        """
        Please provide definition for the following components:
            self.embed_tokens
            self.layers
            self.norm
            self.lm_head
        """
        raise NotImplementedError("init_model() is not implemented")

    def init_inference_optimization(self, config: PretrainedConfig):
        if self.on_device_sampling:
            self.sampler = Sampler(config)

        gqa_sharding_strategy = determine_sharding_strategy(self.tp_degree, self.num_key_value_heads)
        _, num_key_value_heads = get_shardable_head_counts(
            self.tp_degree, self.num_attention_heads, self.num_key_value_heads, gqa_sharding_strategy
        )
        if parallel_state.model_parallel_is_initialized():
            num_kv_heads_per_partition = utils.divide(num_key_value_heads, self.tp_degree)
        else:
            num_kv_heads_per_partition = num_key_value_heads

        hidden_dim_per_head = self.hidden_size // self.num_attention_heads

        self.kv_shape = (
            self.max_batch_size,
            num_kv_heads_per_partition,
            self.max_length,
            hidden_dim_per_head,
        )
        self.past_key_values = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.kv_shape, dtype=config.torch_dtype), requires_grad=False)
                for _ in range(config.num_hidden_layers * 2)
            ]
        )

    def _bucket_slice_kv_cacheline(self, cache):

        if self.padding_side == "right":
            return slice_lhs(cache, self.n_positions, self.SEQ_DIM)
        else:
            max_idx = cache.shape[self.SEQ_DIM]
            return slice_rhs(cache, self.n_positions, max_idx, self.SEQ_DIM)

    def _gather_bucket_slice_into_kv_cacheline(self, idx, bucket_slice):
        max_idx = self.past_key_values[idx].shape[self.SEQ_DIM]
        if self.padding_side == "right":
            remaining = slice_rhs(self.past_key_values[idx], max_idx - self.n_positions, max_idx, self.SEQ_DIM)
            return torch.cat([bucket_slice, remaining], dim=self.SEQ_DIM)
        else:
            remaining = slice_lhs(self.past_key_values[idx], max_idx - self.n_positions, self.SEQ_DIM)
            return torch.cat([remaining, bucket_slice], dim=self.SEQ_DIM)

    def _create_context_attn_mask(self, attention_mask):
        mask = torch.full((self.n_positions, self.n_positions), True, device=attention_mask.device).tril(diagonal=0)
        mask = mask[None, None, :, :].expand(self.batch_size, 1, self.n_positions, self.n_positions)

        if self.padding_side == "right":
            return mask
        else:
            expanded_mask = (
                attention_mask[:, None, None, :]
                .expand(self.batch_size, 1, self.n_positions, self.n_positions)
                .to(torch.bool)
            )
            return torch.logical_and(mask, expanded_mask)

    def _create_simple_attn_mask(self, attention_mask):
        return attention_mask[:, None, None, :].expand(self.batch_size, 1, 1, self.n_positions).to(torch.bool)

    def create_attn_mask(self, attention_mask, is_for_context_encoding, position_ids):
        if is_for_context_encoding:
            return self._create_context_attn_mask(attention_mask)
        else:
            return self._create_simple_attn_mask(attention_mask)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        seq_ids,
        accepted_indices=None,
        current_length=None,
        scatter_index=None,
    ):

        is_for_context_encoding = input_ids.shape[-1] > 1

        # It is either for context encoding or for token generation
        if is_for_context_encoding:
            past_key_values = None
        else:
            past_key_values = []
            for key_layer_idx in range(0, len(self.past_key_values), 2):
                k_cache = self.past_key_values[key_layer_idx]
                v_cache = self.past_key_values[key_layer_idx + 1]
                key_state = self._bucket_slice_kv_cacheline(k_cache)
                value_state = self._bucket_slice_kv_cacheline(v_cache)

                past_key_values.append([key_state, value_state])

        # Prepare attention mask(s)
        attention_mask = self.create_attn_mask(attention_mask, is_for_context_encoding, position_ids)
        active_mask = None

        hidden_states, past_key_values = self.get_model_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            active_mask=active_mask,
        )

        updated_kv_cache = []
        for idx, kv_per_layer in enumerate(past_key_values):
            k_cache = self._bucket_slice_kv_cacheline(self.past_key_values[idx * 2])
            v_cache = self._bucket_slice_kv_cacheline(self.past_key_values[idx * 2 + 1])

            if is_for_context_encoding:
                if self.config.is_continuous_batching:
                    # scatter back to the desired seq_ids
                    seq_id_index_shape = seq_ids.shape[:1] + k_cache.shape[1:]
                    seq_id_index = seq_ids.view(-1, 1, 1, 1).expand(seq_id_index_shape)
                    k_cache = torch.scatter(k_cache, 0, seq_id_index, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 0, seq_id_index, kv_per_layer[1])
                else:
                    # assign back to full kv_cacheline
                    k_cache = kv_per_layer[0]
                    v_cache = kv_per_layer[1]
            else:
                if self.padding_side == "left":
                    # TODO: fix it with scatter after right padding
                    k_cache = k_cache[:, :, 1:, :]
                    v_cache = v_cache[:, :, 1:, :]
                    k_cache = torch.cat([k_cache, kv_per_layer[0]], dim=2)
                    v_cache = torch.cat([v_cache, kv_per_layer[1]], dim=2)
                else:
                    scatter_index_new = position_ids.view(-1, 1, position_ids.shape[-1], 1).expand_as(kv_per_layer[0])
                    k_cache = torch.scatter(k_cache, 2, scatter_index_new, kv_per_layer[0])
                    v_cache = torch.scatter(v_cache, 2, scatter_index_new, kv_per_layer[1])

            k_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2, k_cache)
            v_cache = self._gather_bucket_slice_into_kv_cacheline(idx * 2 + 1, v_cache)

            updated_kv_cache.append(k_cache)
            updated_kv_cache.append(v_cache)

        if self.padding_side == "left":
            index = torch.tensor([hidden_states.shape[1] - 1], device=hidden_states.device)
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)
        else:
            # simple token generation
            index = torch.max(position_ids, dim=1, keepdim=True).indices
            index = index.unsqueeze(1).expand(self.batch_size, 1, self.hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        res = logits
        if self.on_device_sampling:
            # perform sampling on Neuron to get tokens
            res = self.sampler.sample(logits[:, -1, :])

        return [res] + updated_kv_cache

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
    ):
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  # noqa
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # NeuronLlamaModel class manages the KV cache. So the attention_mask will be generated and passed
        # through to LlamaModel. We override the HF's code that generates attention mask because HF does
        # not support left aligned RHS padding. This enables Neuron to achieve higher performance and
        # extensibility.
        #
        # 4d mask is passed through the layers
        # attention_mask = _prepare_4d_causal_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = ()

        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
            )

            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        return (hidden_states, next_decoder_cache)


@dataclass
class CheckPointLoader:
    """Loads model checkpoints

    A Pickable class to be used by compilation processes spawned
    by the ModelBuilder to load checkpoints.

    """

    model_path: Union[str, Path]
    model_prefix: str
    dtype: torch.dtype

    def load_checkpoint(self):
        # this function loads the model's state dictionary and weights from
        # the hf model, removes the prefix and casts to the correct dtype
        model_sd = load_state_dict(self.model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(self.model_prefix):
                updated_param_name = param_name.replace(self.model_prefix, "", 1)
                model_sd[updated_param_name] = model_sd[param_name].to(self.dtype)
                del model_sd[param_name]
            else:
                model_sd[param_name] = model_sd[param_name].to(self.dtype)
        return model_sd


class NeuronBaseForCausalLM(GenerationMixin):
    _STATE_DICT_MODEL_PREFIX = "model."

    _model_cls = None
    _config_cls = None

    # Required by GenerationMixin, but present in PreTrainedModel
    main_input_name = "input_ids"
    _supports_cache_class = False
    # _supports_static_cache = False

    def __init__(self, config: PretrainedConfig, model: torch.jit.ScriptModule):
        super().__init__()

        self.config = config
        self.generation_config = GenerationConfig.from_model_config(config)
        self.vocab_size = config.vocab_size
        self.padding_side = config.padding_side
        self.kv_cache_populated = False

        self.sampler = None

        self.context_encoding_model = ModelWrapper(
            config=self.config,
            model=model,
            tag=CONTEXT_ENCODING_MODEL_TAG,
        )
        self.token_generation_model = ModelWrapper(
            config=self.config,
            model=model,
            tag=TOKEN_GENERATION_MODEL_TAG,
        )

    def can_generate(self):
        # Not needed after transformers 4.50
        return True

    @staticmethod
    def get_compiler_args():
        return "--enable-saturate-infinity --auto-cast=none --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' -O1 "

    @classmethod
    def from_pretrained(cls, model_path: str, config: PretrainedConfig):
        return cls(model_path, config)

    @classmethod
    def export(cls, model_path: Union[str, Path], config: NeuronInferenceConfig, serialize_base_path=None):

        if not os.path.exists(serialize_base_path):
            os.makedirs(serialize_base_path)

        config.save_pretrained(serialize_base_path)
        base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

        checkpoint_loader = CheckPointLoader(model_path, cls._STATE_DICT_MODEL_PREFIX, config.torch_dtype)

        builder = ModelBuilder(
            router=None,
            tp_degree=config.tp_degree,
            checkpoint_loader=checkpoint_loader.load_checkpoint,
            compiler_workdir=base_compile_work_dir,
        )

        # The builder will create a single NxDModel that houses multiple SPMDBucketModel
        # sharing the same weights. Typically, one SPMDBucketModel is created for each
        # input shape that the NxDModel accepts.
        # The SPMDBucketModel used for inference will be selected dynamically at runtime
        # based on the inputs.
        # For LLM models, we typically use different sets of SPMDBucketModel for encoding and
        # token generation, each with its own list of buckets.
        exporters = [
            ContextEncodingModelExporter(cls._model_cls, config),
            TokenGenerationModelExporter(cls._model_cls, config),
        ]
        for exporter in exporters:
            # We need a pickable object to provide the callbacks required by the Builder
            builder.add(
                key=exporter.tag,
                model_instance=exporter.get_model_instance(),
                example_inputs=exporter.input_generator(),
                bucket_config=exporter.bucket_config(),
                compiler_args=cls.get_compiler_args(),
                priority_model_idx=None,
            )

        traced_model = builder.trace(initialize_model_weights=False)
        torch.jit.save(traced_model, NeuronBaseForCausalLM.get_traced_model_path(serialize_base_path))
        del traced_model

        builder.shard_checkpoint(serialize_path=os.path.join(serialize_base_path, "weights/"))

    @staticmethod
    def get_traced_model_path(base_path: Union[str, Path]):
        return os.path.join(base_path, "model.pt")

    @classmethod
    def load(cls, serialize_base_path):

        config = cls._config_cls.from_pretrained(serialize_base_path)

        traced_model = torch.jit.load(NeuronBaseForCausalLM.get_traced_model_path(serialize_base_path))

        SHARD_PREFIX = "tp"
        SHARD_SUFFIX = "_sharded_checkpoint.safetensors"
        sharded_checkpoints_root = os.path.join(serialize_base_path, "weights")
        sharded_checkpoints = glob.glob(SHARD_PREFIX + "*" + SHARD_SUFFIX, root_dir=sharded_checkpoints_root)

        def get_rank(shard):
            return int(shard[len(SHARD_PREFIX) : -len(SHARD_SUFFIX)])

        sharded_checkpoints = sorted(sharded_checkpoints, key=get_rank)
        weights = [load_file(os.path.join(sharded_checkpoints_root, ckpt_path)) for ckpt_path in sharded_checkpoints]

        traced_model.nxd_model.initialize(weights)

        return cls(config, traced_model)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        seq_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        output_attentions, output_hidden_states, return_dict = self._setup_func_config(
            output_attentions, output_hidden_states, return_dict
        )

        # infer attention_mask from position_ids if not provided
        if attention_mask is None:
            attention_mask = self._infer_attention_mask(position_ids)

        self._log_input(input_ids, attention_mask, position_ids, seq_ids)

        if seq_ids is None:
            seq_ids = torch.arange(input_ids.shape[0])

        logits_or_next_tokens = self._get_model_outputs(input_ids, attention_mask, position_ids, seq_ids)

        logging.debug("---output---")
        logging.debug(f"{'tokens' if self.config.on_device_sampling else 'logits'} = %s, ", logits_or_next_tokens)

        return self._construct_output(logits_or_next_tokens)

    def _setup_func_config(self, output_attentions, output_hidden_states, return_dict):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return output_attentions, output_hidden_states, return_dict

    def _infer_attention_mask(self, position_ids):
        assert position_ids is not None, "need to call forward with position_ids if attention_mask is not provided"
        batch_size, seq_len = position_ids.shape
        if position_ids.shape[-1] == 1:
            seq_len = self.config.n_positions
            position_ids_to_compare = position_ids.expand(batch_size, seq_len) - 1
        else:
            seq_len = position_ids.shape[-1]
            position_ids_to_compare = position_ids
        mask = torch.arange(seq_len).view(1, -1).expand(batch_size, seq_len)
        attention_mask = (position_ids_to_compare >= mask).to(dtype=position_ids.dtype)
        return attention_mask

    def _log_input(self, input_ids, attention_mask, position_ids, seq_ids):
        logging.debug("---input---")
        logging.debug("input_ids shape = %s type=%s", input_ids.shape, input_ids.type())
        logging.debug("attention_mask shape = %s type=%s", attention_mask.shape, attention_mask.type())
        logging.debug("position_ids shape = %s type=%s", position_ids.shape, position_ids.type())
        logging.debug("input_ids =%s", input_ids)
        logging.debug("attention_mask =%s", attention_mask)
        logging.debug("position_ids =%s", position_ids)
        logging.debug(f"seq_ids: {seq_ids}")

    def _get_model_outputs(self, input_ids, attention_mask, position_ids, seq_ids):
        if input_ids.shape[-1] > 1:
            outputs = self.context_encoding_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )
            self.kv_cache_populated = True
        else:
            outputs = self.token_generation_model(
                input_ids,
                attention_mask,
                position_ids,
                seq_ids,
            )

        return outputs

    def _construct_output(self, logits_or_next_tokens):
        next_tokens = logits_or_next_tokens

        OutputParams = CausalLMOutputWithPast(
            logits=None if self.config.on_device_sampling else logits_or_next_tokens,
            hidden_states=logits_or_next_tokens,
            attentions=None,
        )

        OutputParams.tokens = next_tokens

        return OutputParams

    # We override this function because we want to change the way attention_mask
    # is updated each iteration.
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_for_token_generation: Optional[bool] = False,
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if is_for_token_generation:
                if self.padding_side == "left":
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
                    attention_mask = attention_mask[:, 1:]
                else:
                    attention_mask = torch.cat(
                        [attention_mask.new_ones((attention_mask.shape[0], 1)), attention_mask], dim=-1
                    )
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if self.kv_cache_populated:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if self.kv_cache_populated:
                position_ids = torch.amax(position_ids, 1, keepdim=True)
                position_ids = position_ids + 1

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", False),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def reset(self):
        # We need to reset the KV cache flag for a new batch of inference.
        # When the flag is reset, the subsequent run will invoke the
        # context encoding model.
        self.kv_cache_populated = False

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        We override the GenerationMixin sample function (_sample for transformers>=4.39.0) to add support for right side padding.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        eos_token_id = 2
        pad_token_id = eos_token_id
        # pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        # eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False
        # auto-regressive generation
        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            is_for_token_generation = self.kv_cache_populated

            # forward pass to get next token
            outputs = self.forward(**model_inputs, return_dict=True)

            if not self.config.on_device_sampling:
                next_token_logits = outputs.logits[:, -1, :]

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                next_token_scores = logits_warper(input_ids, next_token_scores)

                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)

            if not self.config.on_device_sampling:
                if self.sampler is None:
                    self.config.do_sample = True
                    self.sampler = Sampler(self.config)
                next_tokens = self.sampler.sample(outputs.logits[:, -1, :])
            else:
                next_tokens = outputs.tokens

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
                is_for_token_generation=is_for_token_generation,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
            )
        else:
            return input_ids
