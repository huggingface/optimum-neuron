from torch import argmax, count_nonzero, cumsum, gather, nn, rand, subtract, topk
from torch_neuronx.xla_impl.ops import Argmax, Softmax, TopK


class Sampler:
    """
    Use this to implement sampling techniques

    """

    def __init__(self, neuron_config):
        self.on_device_sampling = neuron_config.on_device_sampling
        self.top_k = neuron_config.top_k

    def sample(self, token_logits):
        """
        Function to perform multinomial sampling.

        Input:
            token logits tensor of size (Batch Size, Vocabulary Size)

        Output:
            Tensor containing 1 sampled token id per batch size.
            Output size is (1, Batch Size)

        Note: Using torch.multinomial on device causes trace to hang.
        This is because torch.multinomial performs a number of distribution
        validation steps, which is content dependent. Hence we implement multinomial
        distribution here instead.
        """
        # token_logits has dimensions (batch-size, vocabulary-length)
        # we do all aperations on dim=1
        dim = 1
        keep_dim = False
        num_samples = 1
        if self.top_k == 1:  # do greedy sampling
            if self.on_device_sampling:
                return Argmax.apply(token_logits, dim, keep_dim)  # custom call
            else:
                return argmax(token_logits, dim=dim)
        else:  # do multinomial sampling
            if self.on_device_sampling:  # use TopK and Softmax custom calls
                # 1 is for dim
                top_k_logits = TopK.apply(token_logits, self.top_k, dim)  # custom call
                top_k_logits_values = top_k_logits[0]
                top_k_logits_indices = top_k_logits[1]
                # 1 is for dim
                probs_soft_max = Softmax.apply(top_k_logits_values, dim)  # custom call
            else:  # use torch topk and  softmax
                top_k_logits = topk(input=token_logits, k=self.top_k, dim=dim)
                top_k_logits_indices = top_k_logits.indices
                top_k_logits_values = top_k_logits.values
                probs_soft_max = nn.functional.softmax(input=top_k_logits_values, dim=dim)
            # get CDF
            probs_cumsum = cumsum(input=probs_soft_max, dim=dim)
            # sample 1 value unifromly
            rand_selector = rand((probs_cumsum.shape[0], num_samples), device=token_logits.device)
            # subtract sampled from comulative probs
            diffs = subtract(probs_cumsum, rand_selector)
            # count negative values to find index of sampled value
            counts = count_nonzero((diffs < 0), dim=dim)
            return gather(input=top_k_logits_indices, dim=dim, index=counts.unsqueeze(1)).flatten()
