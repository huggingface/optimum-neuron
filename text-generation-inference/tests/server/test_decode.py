from helpers import create_request
from text_generation_server.generator import NeuronGenerator
from text_generation_server.pb.generate_pb2 import Batch


def test_decode(neuron_model_config):
    """Verify that a decoding for a single request generates the expected output."""
    config_name = neuron_model_config["name"]
    neuron_model_path = neuron_model_config["neuron_model_path"]
    generator = NeuronGenerator.from_pretrained(neuron_model_path)
    for do_sample in [True, False]:
        mode = "sample" if do_sample else "greedy"
        print(f"{config_name}[{mode}]")
        _test_decode(config_name, generator, do_sample)
        generator.clear()


def _test_decode(config_name, generator, do_sample):
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    max_new_tokens = 20
    request = create_request(id=0, inputs=input_text, max_new_tokens=max_new_tokens, do_sample=do_sample)
    max_length = generator.model.max_length
    batch = Batch(id=0, requests=[request], size=1, max_tokens=max_length)
    generations, next_batch = generator.prefill(batch)
    # We already generated one token: call decode max_new_tokens - 1 times
    for _ in range(max_new_tokens - 1):
        assert next_batch.size == 1
        assert next_batch.max_tokens == max_length
        assert len(generations) == 1
        assert len(generations[0].tokens.ids) == 1
        generations, next_batch = generator.decode([next_batch])
    assert next_batch is None
    assert len(generations) == 1
    output = generations[0].generated_text
    assert output.generated_tokens == max_new_tokens
    assert output.finish_reason == 0
    if do_sample:
        expected_text = {
            "gpt2": " The sun was set just three miles south of the city. I had just watched a big fireworks display",
            "llama": " In the corner booth of O'Malley's Pub sat two old friends, retired police officer",
            "mistral": " The sun was out and there was an unusual amount of light, so I wandered along the",
        }[config_name]
    else:
        expected_text = {
            "gpt2": '\n\n"I\'m going to go to bed," I said.\n\n"I\'m going',
            "llama": " In the small town of Meadowgrove, everyone knew each other, and they all took",
            "mistral": "\nThe clocks were striking thirteen.\nThe clocks were striking thirteen.",
        }[config_name]
    assert output.text == expected_text
