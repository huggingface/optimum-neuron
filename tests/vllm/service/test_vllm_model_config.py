import re
from tempfile import TemporaryDirectory

import pytest
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from vllm.config.model import ModelConfig
from vllm.config.parallel import ParallelConfig

from optimum.neuron import NeuronModelForCausalLM


@pytest.mark.asyncio
async def test_vllm_accepts_model_config_tp_not_dividing_num_attention_heads(vllm_launcher):
    """Test that when using the Neuron platform, vLLM accepts a model configuration
    where the tensor parallel size does not divide the number of attention heads.
    This is important for Neuron models like Llama 4 Scout 17B with TP=32."""
    num_attention_heads = 5
    tensor_parallel_size = 2

    # Create a Llama model config where the number of attention heads is not even
    llama_config = LlamaConfig(
        hidden_size=128 * num_attention_heads,
        intermediate_size=256,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        num_hidden_layers=1,
        max_position_embeddings=512,
    )
    # Instantiate and save the model and its tokenizer to a temporary directory
    llama_model = LlamaForCausalLM(llama_config)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.1-8B")
    with TemporaryDirectory() as tmp_dir:
        llama_model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        # Export the model to neuron format
        neuron_config = NeuronModelForCausalLM.get_neuron_config(tmp_dir, tensor_parallel_size=tensor_parallel_size)
        neuron_llama_model = NeuronModelForCausalLM.export(tmp_dir, neuron_config=neuron_config, config=llama_config)
        neuron_llama_model.save_pretrained(tmp_dir)

        # First, verify that an error is raised when checking the model config manually
        vllm_model_config = ModelConfig(tmp_dir)
        parallel_config = ParallelConfig(tensor_parallel_size=tensor_parallel_size)
        error_msg = f"Total number of attention heads ({num_attention_heads}) must be divisible by tensor parallel size ({tensor_parallel_size})."
        with pytest.raises(ValueError, match=re.escape(error_msg)):
            vllm_model_config.verify_with_parallel_config(parallel_config)

        # Now, launch a vLLM service with the model and verify it starts successfully
        with vllm_launcher("test_service", tmp_dir) as vllm_service:
            await vllm_service.health(600)
