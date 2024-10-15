import torch
from llama2.neuron_modeling_llama import (
    NeuronLlamaConfig,
    NeuronLlamaForCausalLM,
    NeuronLlamaModel,
)
from runner import InferenceRunner
from transformers import AutoTokenizer

from neuronx_distributed.parallel_layers.checkpointing import _invoke_preshard_hook
from neuronx_distributed.quantization.quantization_config import QuantizationType
from neuronx_distributed.quantization.quantization_utils import (
    quantize_pytorch_model_per_channel_symmetric,
    quantize_pytorch_model_per_tensor_symmetric,
)


class LlamaRunner(InferenceRunner):
    def load_hf_model(self):
        return NeuronLlamaForCausalLM.load_hf_model(self.model_path)

    def load_neuron_model_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        self.config = self.get_config_for_nxd(
            batch_size,
            1,
            max_prompt_length=max_prompt_length,
            sequence_length=sequence_length,
            enable_bucketing=False,
        **kwargs)
        self.config.torch_dtype = torch.float32

        neuron_model = NeuronLlamaModel(self.config)

        state_dict = NeuronLlamaForCausalLM.get_state_dict(self.model_path, config=self.config)
        _invoke_preshard_hook(neuron_model, state_dict)

        neuron_model.load_state_dict(state_dict, strict=False)

        if self.config.torch_dtype == torch.bfloat16:
            neuron_model.bfloat16()

        model = NeuronLlamaForCausalLM(None, self.config)
        model.context_encoding_model.model = neuron_model
        model.token_generation_model.model = neuron_model
        return model

    def generate_quantized_hf_checkpoints_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        config = self.get_config_for_nxd(batch_size, 1, max_prompt_length, sequence_length, **kwargs)
        config.torch_dtype = torch.float32

        quantized_state_dict = NeuronLlamaForCausalLM.generate_quantized_state_dict(
            model_path=self.model_path, config=config
        )
        return quantized_state_dict

    def load_quantized_neuron_model_on_cpu(self, max_prompt_length, sequence_length, batch_size, **kwargs):
        model = self.load_neuron_model_on_cpu(max_prompt_length, sequence_length, batch_size, **kwargs)

        quantization_type = QuantizationType(kwargs.get("quantization_type", "per_tensor_symmetric"))
        if quantization_type == QuantizationType.PER_TENSOR_SYMMETRIC:
            return quantize_pytorch_model_per_tensor_symmetric(model, inplace=True)
        elif quantization_type == QuantizationType.PER_CHANNEL_SYMMETRIC:
            return quantize_pytorch_model_per_channel_symmetric(model, inplace=True)
        else:
            raise RuntimeError(f"quantization_type: {quantization_type} not supported")

    def load_neuron_model(self, traced_model_path):
        config = NeuronLlamaConfig.from_pretrained(traced_model_path)
        model = NeuronLlamaForCausalLM.from_pretrained("", config)
        self.config = config

        model.load(traced_model_path)
        if config.torch_dtype == torch.bfloat16:
            model.bfloat16()

        return model

    def load_tokenizer(self, padding_side=None):
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.pad_token_id = self.config.eos_token_id
        tokenizer.padding_side = padding_side if padding_side else self.get_padding_side()
        return tokenizer

    def get_config_cls(self):
        return NeuronLlamaConfig

    def get_model_cls(self):
        return NeuronLlamaForCausalLM

    def get_padding_side(self):
        return "right"

    def get_default_hf_generation_config_kwargs(self):
        config = super().get_default_hf_generation_config_kwargs()
        # set to eos_token_id as that's done in load_tokenizer
        config['pad_token_id'] = self.generation_config.eos_token_id

        return config


if __name__ == "__main__":
    LlamaRunner.cmd_execute()
