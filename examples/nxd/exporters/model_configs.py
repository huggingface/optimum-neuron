import torch
from models.llama.modeling_llama import NeuronLlamaModel
from modules.autobucketing import generate_buckets, get_context_encoder_bk, get_token_generation_bk
from modules.config import NeuronInferenceConfig
from neuronx_distributed.trace.model_builder import BaseModelInstance
from torch_neuronx import BucketModelConfig

from optimum.exporters.base import ExportConfig


# This is a very simplified version of optimum TaskManager used for demonstration only
_EXPORT_CONFIGS = {}


def get_exporter_config_constructor(model_type: str):
    return _EXPORT_CONFIGS[model_type]


def register_export_config(model_type):
    def wrapper(cls):
        _EXPORT_CONFIGS[model_type] = cls
        return cls

    return wrapper


class DecoderModelInstance(BaseModelInstance):
    # This class could probably be replaced by a callable, like it is done in optimum-neuron

    def __init__(self, model_cls, config, buckets):
        self.model_cls = model_cls
        self.module = None
        self.input_output_aliases = None
        self.config = config
        self.buckets = buckets

    def load_module(self):
        float_model = self.model_cls(self.config)
        float_model.eval()

        if self.config.torch_dtype == torch.bfloat16:
            float_model.bfloat16()

        self.module = float_model

    def get(self, bucket_rank, **kwargs):
        if bucket_rank is not None:
            self.module.n_positions = self.buckets[bucket_rank]

        # Currently we have to init an input_output_aliases map for
        # each buckets, otherwise it will fail the aliasing setup when
        # generating HLO
        self.input_output_aliases = {}
        num_output_from_trace = 1
        for i in range(len(self.module.past_key_values)):
            self.input_output_aliases[self.module.past_key_values[i]] = num_output_from_trace + i
        return self.module, self.input_output_aliases


@register_export_config("llama")
class LlamaNeuronExportConfig(ExportConfig):
    # This class would typically be called LlamaNeuronConfig in optimum-neuron
    # The different naming is to avoid confusion with the pretrained config class in NxD examples

    _STATE_DICT_MODEL_PREFIX = "model."
    _MODEL_CLS = NeuronLlamaModel
    _ATTN_CLS = "NeuronLlamaAttention"

    def __init__(self, config: NeuronInferenceConfig, is_prefill: bool):
        self.is_prefill = is_prefill
        self.config = config
        self.max_input_tokens = config.max_context_length if is_prefill else 1
        if is_prefill:
            if config.enable_bucketing:
                buckets = generate_buckets(128, config.max_context_length)
            else:
                buckets = [config.max_context_length]
        else:
            if config.enable_bucketing:
                buckets = generate_buckets(128, config.max_length)
            else:
                buckets = [config.max_length]
        self.buckets = buckets

    @staticmethod
    def get_compiler_args():
        return "--enable-saturate-infinity --auto-cast=none --model-type=transformer --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' -O1 "

    def input_generator(self):
        inputs = []
        for bucket in self.buckets:
            batch_size = self.config.batch_size
            n_active_tokens = min(self.max_input_tokens, bucket)

            input_ids = torch.zeros((batch_size, n_active_tokens), dtype=torch.int64)
            attention_mask = torch.zeros((batch_size, bucket), dtype=torch.int64)
            position_ids = torch.zeros((batch_size, n_active_tokens), dtype=torch.int64)
            seq_ids = torch.zeros((batch_size), dtype=torch.int64)

            inputs.append((input_ids, attention_mask, position_ids, seq_ids))

        return inputs

    def get_model_instance(self):
        return DecoderModelInstance(model_cls=self._MODEL_CLS, config=self.config, buckets=self.buckets)

    def bucket_config(self):
        if not self.config.enable_bucketing:
            return None
        bucket_degree = len(self.buckets)
        if self.is_prefill:
            return BucketModelConfig(
                bucket_kernel=get_context_encoder_bk,
                bucket_kernel_constant_args=(
                    torch.tensor(self.buckets),
                    self.config.padding_side,
                    self.config.pad_token_id,
                ),
                shared_state_buffer=None,
                func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
            )
        else:
            return BucketModelConfig(
                bucket_kernel=get_token_generation_bk,
                bucket_kernel_constant_args=(torch.tensor(self.buckets), self.config.padding_side),
                shared_state_buffer=None,
                func_kwargs=[{"bucket_rank": i} for i in range(bucket_degree)],
            )
