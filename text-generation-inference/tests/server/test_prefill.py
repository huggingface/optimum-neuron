from helpers import create_request
from text_generation_server.generator import NeuronGenerator
from text_generation_server.pb.generate_pb2 import Batch


def test_prefill(neuron_model_config):
    """Verify that a prefill for a single request generates the expected output."""
    config_name = neuron_model_config["name"]
    neuron_model_path = neuron_model_config["neuron_model_path"]
    generator = NeuronGenerator.from_pretrained(neuron_model_path)
    max_batch_size = 4
    assert generator.model.batch_size >= max_batch_size
    for num_requests in [1, max_batch_size]:
        for do_sample in [True, False]:
            mode = "sample" if do_sample else "greedy"
            print(f"[{mode}]: {num_requests} requests")
            _test_prefill(config_name, generator, num_requests, do_sample)
            generator.clear()


def _test_prefill(config_name, generator, batch_size, do_sample):
    requests = []
    max_new_tokens = 20
    input_text = "It was a bright cold day in April, and the clocks were striking thirteen."
    for i in range(batch_size):
        requests.append(create_request(id=i, inputs=input_text, do_sample=do_sample, max_new_tokens=max_new_tokens))
    # Let's be pessimistic when estimating max_tokens
    max_length = generator.model.max_length
    batch = Batch(id=0, requests=requests, size=batch_size, max_tokens=batch_size * max_length)
    generations, next_batch = generator.prefill(batch)
    assert next_batch.size == batch_size
    # Whatever was passed as max_tokens, the server will correct it
    # because of static batching
    assert next_batch.max_tokens == batch_size * max_length
    assert len(generations) == batch_size
    if do_sample:
        expectations = {"gpt2": [383, " The"], "llama": [560, " In"], "mistral": [450, " The"]}[config_name]
    else:
        expectations = {"gpt2": [198, "\n"], "llama": [560, " In"], "mistral": [13, "\n"]}[config_name]
    for g in generations:
        tokens = g.tokens
        assert tokens.ids[0] == expectations[0]
        assert tokens.texts[0] == expectations[1]
