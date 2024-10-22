from runner import InferenceRunner

from llama2.neuron_modeling_llama import (
    NeuronLlamaConfig,
    NeuronLlamaForCausalLM,
)


class LlamaRunner(InferenceRunner):

    def get_config_cls(self):
        return NeuronLlamaConfig

    def get_model_cls(self):
        return NeuronLlamaForCausalLM

    def get_default_hf_generation_config_kwargs(self):
        config = super().get_default_hf_generation_config_kwargs()
        # set to eos_token_id as that's done in load_tokenizer
        config["pad_token_id"] = self.generation_config.eos_token_id

        return config


if __name__ == "__main__":
    LlamaRunner.cmd_execute()
