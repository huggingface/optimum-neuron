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
This is a simple command-line chat application that has contextual memory.
"""

from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM

# Load the model compiled for AWS Neuron
model = NeuronModelForCausalLM.from_pretrained("./mistral_neuron", local_files_only=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("./mistral_neuron")
tokenizer.pad_token_id = tokenizer.eos_token_id

def format_chat_prompt(message, history, max_tokens):
    """Formats an entire chat history to enable contextual memory."""
    chat = []

    # Add each former interaction to the chat list with alternating roles, user and assistant.
    for interaction in history:
        chat.append({"role": "user", "content": interaction[0]})
        chat.append({"role": "assistant", "content": interaction[1]})

    # Add the new (user) message to the chat flow.
    chat.append({"role": "user", "content": message})

    # Apply the chat template to each chat message and ensure we do not exceed max_tokens
    for i in range(0, len(chat), 2):
        # apply the chat message to every pair of messages from user and assistant up to i
        prompt = tokenizer.apply_chat_template(chat[i:], tokenize=False)

        # Validate that our response does not exceed max_tokens.
        # If it does, the for loop truncates the first message from the prompt until we are under max_tokens.
        # This way, we never pass more than the alloted max_tokens to the model. 
        tokens = tokenizer(prompt) 
        if len(tokens.input_ids) <= max_tokens:
            return prompt

    # If we've exceeded max_tokens, raise SystemError.
    # This shouldn't be reached unless something goes wrong, such as the
    # initial message and subsequent response exceeding the token limit.
    raise SystemError

def chat(history, max_tokens):
    """This function runs recursively to take user input, making the chat bot functional."""
    # Take input from user
    message = input("Enter input: ")

    # Stop the program if the user types "quit"
    if message == "quit":
        return

    # Tokenize the formatted prompt
    inputs = tokenizer(format_chat_prompt(message, history, max_tokens), return_tensors="pt")

    # Do inference to generate a response
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.9
    )

    # Decode the response to a string, and remove the prompt.
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    # Print the response
    print(response)

    # Add the message and response to history 
    history.append([message, response])

    # Repeat
    chat(history, max_tokens)

if __name__ == "__main__":
    # Define an empty history and max number of tokens
    history = []
    max_tokens = 4096

    chat(history, max_tokens)
