from transformers import AutoTokenizer

from optimum.neuron.utils.import_utils import is_package_available


if is_package_available("datasets"):
    from datasets import load_dataset


def get_long_prompt(model_id, min_tokens, max_tokens):
    """Get a long prompt for testing purposes.

    The prompt is selected from the 'fka/awesome-chatgpt-prompts' dataset
    based on the number of tokens it contains.

    Args:
        model_id (str): The model identifier to use for tokenization.
        min_tokens (int): Minimum number of tokens the prompt should have.
        max_tokens (int): Maximum number of tokens the prompt should have.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    for item in dataset:
        prompt = item["prompt"]
        tokens = tokenizer(prompt, return_tensors="pt")
        num_tokens = tokens["input_ids"].shape[1]
        if min_tokens <= num_tokens <= max_tokens:
            return prompt
    raise ValueError(f"No prompt found with tokens between {min_tokens} and {max_tokens}.")
