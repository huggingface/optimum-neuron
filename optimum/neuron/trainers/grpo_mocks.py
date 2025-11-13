# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
Mock implementations for GRPO trainer testing and development.

This module provides mock implementations of vLLM client and other components
to enable development and testing of NeuronGRPOTrainer without requiring a full
vLLM server setup.
"""

from optimum.utils import logging


logger = logging.get_logger()


class MockVLLMClient:
    """
    Mock vLLM client that generates dummy completions for testing.

    This mock client simulates the behavior of a real vLLM server by generating
    placeholder completions. It's useful for:
    - Development without vLLM server setup
    - Testing trainer logic independently of generation quality
    - Unit testing GRPO training loop

    Args:
        tokenizer: Tokenizer to use for encoding/decoding
        max_completion_length: Maximum length of generated completions

    Note:
        This is a development tool and should not be used in production.
        Generated completions are deterministic placeholders, not real language model outputs.
    """

    def __init__(self, tokenizer, max_completion_length=256):
        self.tokenizer = tokenizer
        self.max_completion_length = max_completion_length
        logger.warning(
            "Using MockVLLMClient for development. This generates placeholder completions "
            "and should only be used for testing and development."
        )

    def generate(
        self,
        prompts: list[str],
        images=None,
        n: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 256,
        repetition_penalty: float = 1.0,
        truncate_prompt_tokens=None,
        guided_decoding_regex=None,
        generation_kwargs=None,
    ):
        """
        Generate mock completions for the given prompts.

        Args:
            prompts: List of prompt strings
            images: Optional list of images (not used in mock)
            n: Number of completions to generate per prompt
            temperature: Sampling temperature (not used in mock)
            top_p: Nucleus sampling parameter (not used in mock)
            top_k: Top-k sampling parameter (not used in mock)
            min_p: Minimum probability threshold (not used in mock)
            max_tokens: Maximum tokens to generate
            repetition_penalty: Repetition penalty (not used in mock)
            truncate_prompt_tokens: Maximum prompt length
            guided_decoding_regex: Regex for guided decoding (not used in mock)
            generation_kwargs: Additional generation arguments (not used in mock)

        Returns:
            Dictionary with keys:
            - prompt_ids: List of tokenized prompts (one per prompt)
            - completion_ids: List of tokenized completions (n per prompt)
            - logprobs: List of log probabilities (one list per completion)
        """
        prompt_ids = []
        completion_ids = []
        logprobs = []

        for prompt in prompts:
            # Tokenize prompt
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

            # Truncate if needed
            if truncate_prompt_tokens is not None and len(prompt_tokens) > truncate_prompt_tokens:
                prompt_tokens = prompt_tokens[-truncate_prompt_tokens:]

            prompt_ids.append(prompt_tokens)

            # Generate n completions per prompt
            for i in range(n):
                # Generate mock completion
                # Use a simple pattern: repeat EOS token to create fixed-length completion
                # In real scenario, this would be actual LLM generation
                completion_length = min(max_tokens, self.max_completion_length)

                # Generate completion: cycle through safe token IDs
                completion = [self.tokenizer.eos_token_id] * completion_length
                completion_ids.append(completion)

                # Generate mock logprobs (uniform negative values)
                # Real logprobs would come from the model's probability distribution
                completion_logprobs = [-1.0] * completion_length
                logprobs.append(completion_logprobs)

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
        }

    def init_communicator(self, device):
        """
        Mock initialization of communicator.

        Args:
            device: Device to initialize on (not used in mock)
        """
        pass

    def update_named_param(self, name, data):
        """
        Mock update of named parameter.

        In a real vLLM setup, this would sync model weights to the vLLM server.
        For mock mode, this is a no-op since we're not using a real server.

        Args:
            name: Parameter name
            data: Parameter data tensor (not used in mock)
        """
        pass

    def reset_prefix_cache(self):
        """
        Mock reset of prefix cache.

        In a real vLLM setup, this would clear the KV cache for prefix caching.
        For mock mode, this is a no-op since we're not using a real server.
        """
        pass


def create_mock_vllm_client(tokenizer, args):
    """
    Factory function to create a mock vLLM client.

    Args:
        tokenizer: Tokenizer to use for the mock client
        args: Training arguments containing max_completion_length

    Returns:
        MockVLLMClient instance
    """
    return MockVLLMClient(
        tokenizer=tokenizer,
        max_completion_length=args.max_completion_length,
    )
