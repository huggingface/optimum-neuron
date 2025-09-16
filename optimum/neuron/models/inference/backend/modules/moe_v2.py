import torch
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.moe.shared_experts import SharedExperts


def initialize_moe_module(
    config,
    neuron_config,
    router_dtype=torch.float32,
    router_act_fn="sigmoid",
    n_shared_experts: int | None = None,
    fused_shared_experts: bool = False,
    early_expert_affinity_modulation: bool = False,
):
    """
    Initializes and returns an MoE module corresponding to the given configuration.
    """

    router = RouterTopK(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=router_dtype,
        act_fn=router_act_fn,
    )
    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,
            glu_mlp=neuron_config.glu_mlp,
            early_expert_affinity_modulation=early_expert_affinity_modulation,
        ),
        dtype=neuron_config.torch_dtype,
    )
    shared_experts = None
    if n_shared_experts is not None:
        shared_experts = SharedExperts(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_shared_experts=n_shared_experts,
            hidden_act=config.hidden_act,
            dtype=neuron_config.torch_dtype,
            reduce_dtype=neuron_config.torch_dtype,
            fused_gate_up_projection=fused_shared_experts,
        )

    moe = MoE(router=router, expert_mlps=expert_mlps, shared_experts=shared_experts)
    # Set MoE module in eval mode
    moe.eval()
    return moe
