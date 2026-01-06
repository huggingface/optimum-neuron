import torch
from vllm.config.model import LogprobsMode
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from ..models.inference.backend.config import NeuronConfig
from ..models.inference.backend.modules.generation.sampling import Sampler as NeuronTopkToppSampler


_SAMPLING_EPS = 1e-5


class NeuronSampler(Sampler):
    """
    A sampler class optimized for AWS Neuron.

    This class extends the base Sampler class from vLLM and is tailored to work efficiently
    with AWS Neuron hardware.
    The main functionality is to perform sampling using a Neuron-optimized top-k and top-p sampler.
    """

    def __init__(self, neuron_config: NeuronConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk_topp_sampler = NeuronTopkToppSampler(neuron_config=neuron_config, do_sample=True, on_cpu=True)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                return greedy_sampled, None

        # Apply logits processors that only apply to random sampling
        # (argmax invariant)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        # Apply temperature, top_k and/or top_p.
        sampling_params_tensor = torch.cat(
            [
                sampling_metadata.top_k.unsqueeze(-1),
                sampling_metadata.top_p.unsqueeze(-1),
                sampling_metadata.temperature.unsqueeze(-1),
            ],
            dim=-1,
        )
        random_sampled = self.topk_topp_sampler(logits, sampling_params_tensor)

        if greedy_sampled is None:
            return random_sampled, None

        sampled = torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled,
            random_sampled,
            out=greedy_sampled,  # Reuse tensor
        )
        return sampled, None
