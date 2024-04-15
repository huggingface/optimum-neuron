# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import torch
from transformers import AutoTokenizer

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
def test_generation_llama_padded_inputs():
    model_id = "NousResearch/Llama-2-7b-chat-hf"
    model_kwargs = {"batch_size": 4, "sequence_length": 4096, "auto_cast_type": "f16", "num_cores": 2}
    model = NeuronModelForCausalLM.from_pretrained(model_id, export=True, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt = "One of my fondest memory is of my grandmother making homemade bread"
    first_input = tokenizer(prompt)
    first_ids = first_input["input_ids"]
    first_mask = first_input["attention_mask"]
    max_padding = 12
    input_len = len(first_ids)
    for i in range(max_padding):
        second_ids = [tokenizer.eos_token_id] * i + first_ids[: input_len - i]
        second_mask = [0] * i + [1] * (input_len - i)
        input_ids = torch.tensor([first_ids, second_ids], dtype=torch.int64)
        attention_mask = torch.tensor([first_mask, second_mask], dtype=torch.int64)
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=10
        )
        # Verify we did not generate any unknown token
        assert torch.all(outputs[:, -1] != 0)
