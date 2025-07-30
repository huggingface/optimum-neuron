# Adapted from https://github.com/aws-neuron/neuronx-distributed/blob/c9ee222866916e2f2ab10be884a7ad237c7cb7f4/src/neuronx_distributed/modules/moe/expert_mlps.py
from typing import Any, Callable, Optional, Union

import torch
from neuronx_distributed.modules.moe.model_utils import DEFAULT_BLOCK_SIZE, DEFAULT_LNC_SIZE
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig, RoutedExpertsMLPOpsConfig
from neuronx_distributed.utils.logger import get_logger
from neuronxcc.nki._private_kernels.blockwise_mm import BlockShardStrategy
from torch.distributed import ProcessGroup

from .expert_mlps_v2 import ExpertMLPsV2


logger = get_logger()


class ExpertMLPs(ExpertMLPsV2):
    """Class which obtains the output from passing the token hidden states through the assigned expert(s).
    Arguments:
        num_experts: Total number of experts.
        top_k: Number of experts activated per token. Should be less than or equal to num_experts.
        hidden_size: Hidden dimension.
        intermediate_size: Intermediate dimension used in the MLPs.
        hidden_act: Activation function. See ACT2FN for supported activations.
        glu_mlp: Whether to use the Gated Linear Unit in the MLP. If True, then a combination of gate and up projection is performed in the MLP.
                 Otherwise, a simple up projection is performed.
        capacity_factor: Hyperparameter which controls the expert capacity, and determines the rate of token dropping.
                         If None, then assumed to be running with 'full capacity' (i.e. no tokens dropped).
        block_size: block size used for blockwise matmul
        normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
                                    Should be used only with top_k > 1.
        return_bias: Whether to return the bias in the forward pass. Currently not supported.
        init_method: Function used for initializing the gate and up projection linear layer weights.
        output_layer_init_method: Function used for initializing the down projection linear layer weights.
        dtype: Datatype for the layer weights.
        device: Device for the layer weights.
        enable_spmd_rank: use rank information available at runtime in inference i.e., get tp_rank from global rank
        blockwise_nki_autograd_cls: NKI function that implements blockwise matmul for expert MLPs which will default to BlockwiseMatmulNKIFunc
                                    when specified None. Currently only BlockwiseMatmulNKIFunc is supported.

        use_torch_block_wise: Force using torch implementation of blockwise matmul for expert MLPs instead of invoking NKI kernel.
        logical_nc_config: lnc_size (1 or 2). Default to 1 on trn1, and 2 on trn2.
        parallelize_token_to_block_mapping: parallel computation of block position to token indices mapping. Enabled by default, can be disabled in testing
                                            to rule out collectives issue.
        early_expert_affinity_modulation: scale the inputs to experts by expert affinities before going though expert MLP computation, then post scale the expert outputs
        optimized_block_to_token_mapping: If enabled, token position in blocks will only include top k experts.
        use_block_parallel: Enable calling block parallel blockwise matmuk nki kernel
        block_sharding_strategy: corresponds to different block parallel blockwise matmul kernel
        skip_dma: kernel optimizations for skip tokens and skip weights. When skip token is true, inputs to blockwise kernel do not need to be padded.
                  always_augment_inputs_for_blockwise_matmul: always pad the inputs to blockwise kernel regardless of the value of skip dma.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        glu_mlp: bool,
        capacity_factor: Union[None, float],
        block_size: Union[None, int] = None,
        normalize_top_k_affinities: bool = False,
        return_bias: bool = False,
        init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
        output_layer_init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        expert_model_parallel_group: Optional[ProcessGroup] = None,
        enable_spmd_rank: bool = False,  # spmd_rank will be removed once we support ReplicaID (P87857655)
        blockwise_nki_autograd_cls=None,
        use_torch_block_wise: bool = False,
        logical_nc_config=DEFAULT_LNC_SIZE,  # uses lnc1 blockwise kernel by default
        parallelize_token_to_block_mapping: bool = True,
        early_expert_affinity_modulation: bool = False,
        optimized_block_to_token_mapping: bool = True,
        use_block_parallel: bool = False,
        always_augment_inputs_for_blockwise_matmul: bool = False,
        block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.HI_LO,
        skip_dma_token: bool = False,
        skip_dma_weight: bool = False,
    ):
        routed_experts_mlp_config = RoutedExpertsMLPOpsConfig(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            glu_mlp=glu_mlp,
            normalize_top_k_affinities=normalize_top_k_affinities,
            early_expert_affinity_modulation=early_expert_affinity_modulation,
            input_layer_init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            capacity_factor=capacity_factor,
            top_k=top_k,
            hidden_act=hidden_act,
            enable_spmd_rank=enable_spmd_rank,
        )

        blockwise_matmul_config = BlockwiseMatmulConfig.from_kwargs(
            block_size=block_size if block_size else DEFAULT_BLOCK_SIZE,
            logical_nc_config=logical_nc_config,
            use_torch_block_wise=use_torch_block_wise,
            blockwise_nki_autograd_cls=blockwise_nki_autograd_cls,
            parallelize_token_to_block_mapping=parallelize_token_to_block_mapping,
            early_expert_affinity_modulation=early_expert_affinity_modulation,
            optimized_block_to_token_mapping=optimized_block_to_token_mapping,
            use_block_parallel=use_block_parallel,
            always_augment_inputs_for_blockwise_matmul=always_augment_inputs_for_blockwise_matmul,
            block_sharding_strategy=block_sharding_strategy,
            skip_dma_token=skip_dma_token,
            skip_dma_weight=skip_dma_weight,
        )

        super().__init__(
            routed_experts_mlp_config=routed_experts_mlp_config,
            blockwise_matmul_config=blockwise_matmul_config,
            dtype=dtype,
            device=device,
            return_bias=return_bias,
            tensor_model_parallel_group=tensor_model_parallel_group,
            expert_model_parallel_group=expert_model_parallel_group,
        )
