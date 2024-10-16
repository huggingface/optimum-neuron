import torch
from runner import InferenceRunner
from transformers import AutoTokenizer

from llama2.neuron_modeling_llama import (
    NeuronLlamaConfig,
    NeuronLlamaForCausalLM,
)


class LlamaRunner(InferenceRunner):

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
        config["pad_token_id"] = self.generation_config.eos_token_id

        return config


if __name__ == "__main__":
    LlamaRunner.cmd_execute()
