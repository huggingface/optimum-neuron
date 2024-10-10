#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team and Amazon Web Services, Inc. All rights reserved.
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
"""
This script demonstrates the results of fine-tuning on the gsm8k dataset by
generating a response to a grade school math question taken from the dataset.
"""

from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM

# Load the compiled model.
model = NeuronModelForCausalLM.from_pretrained("./mistral_neuron", local_files_only=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./mistral_neuron")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Set a message to send to the model for inferencing
message = f"[INST] Two girls each got 1/6 of the 24 liters of water. Then a boy got 6 liters of water. How many liters of water were left? [/INST]\n\n"
tokenized_message = tokenizer(message, return_tensors="pt")

# Do the inferencing
outputs = model.generate(
    **tokenized_message,
    max_new_tokens=512, # How many tokens the model can generate in the response
    do_sample=True, # Use sampling or greedy decoding
    temperature=0.9, # The value used to modulate next token probabilities
    top_k=50, # The number of highest probability vocabulary tokens to keep for top-k-filtering
    top_p=0.9 # If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
)

# Decode the output from a tensor array to text.
answer = tokenizer.decode(outputs[0][len(tokenized_message[0]):], skip_special_tokens=True)

print(answer)
