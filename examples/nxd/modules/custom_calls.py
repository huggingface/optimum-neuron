from torch import nn, ones, float32, rsqrt
from torch_neuronx.xla_impl.ops import RmsNorm
from neuronx_distributed.parallel_layers import parallel_state

class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return RmsNorm.apply(hidden_states, self.weight, self.variance_epsilon, 2)
