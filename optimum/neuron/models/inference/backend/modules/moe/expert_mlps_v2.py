# Adapted from https://github.com/aws-neuron/neuronx-distributed/blob/c9ee222866916e2f2ab10be884a7ad237c7cb7f4/src/neuronx_distributed/modules/moe/expert_mlps_v2.py
import math
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
from neuronx_distributed.modules.moe.blockwise import (
    BlockwiseMatmulNKIFunc,
    ExpertAffinityScaleMode,
    SkipMode,
    TorchBlockwiseTraining,
    augment_inputs_for_padded_blockwise_matmul,
    can_use_blockwise_matmul_nki,
)
from neuronx_distributed.modules.moe.model_utils import ACT2FN, DEFAULT_SELECTIVE_LOADING_THRESHOLD
from neuronx_distributed.modules.moe.moe_configs import BlockwiseMatmulConfig, RoutedExpertsMLPOpsConfig
from neuronx_distributed.parallel_layers import mappings, parallel_state
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import (
    get_expert_model_parallel_group,
    get_expert_model_parallel_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
    rmsg,
)
from neuronx_distributed.utils.logger import get_logger
from neuronx_distributed.utils.tensor_utils import cumsum
from torch.distributed import ProcessGroup

from .experts import Experts


logger = get_logger()


class ExpertMLPsV2(torch.nn.Module):
    """Class which obtains the output from passing the token hidden states through the assigned expert(s).

    Arguments:
        routed_experts_mlp_config: routed expert configs. Details are in neuronx_distributed.modules.moe.model_utils
        blockwise_matmul_config: blockwise matmul configs. Details are in neuronx_distributed.modules.moe.model_utils
        return_bias: Whether to return the bias in the forward pass. Currently not supported.
        init_method: Function used for initializing the gate and up projection linear layer weights.
        dtype: Datatype for the layer weights.
        device: Device for the layer weights.
    """

    def __init__(
        self,
        routed_experts_mlp_config: RoutedExpertsMLPOpsConfig,
        blockwise_matmul_config: BlockwiseMatmulConfig = BlockwiseMatmulConfig.default(),
        return_bias: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        tensor_model_parallel_group: Optional[ProcessGroup] = None,
        expert_model_parallel_group: Optional[ProcessGroup] = None,
        glu_activation_fn: Optional[Callable[..., Any]] = None,
        # spmd_rank will be removed once we support ReplicaID (P87857655)
    ):
        super().__init__()
        self.routed_experts_mlp_config = routed_experts_mlp_config
        self.blockwise_matmul_config = blockwise_matmul_config
        skip_dma = SkipMode(blockwise_matmul_config.skip_dma_token, blockwise_matmul_config.skip_dma_weight)
        setattr(self.blockwise_matmul_config, "skip_dma", skip_dma)
        self.validate_routed_experts_configs(routed_experts_mlp_config)
        if return_bias:
            raise NotImplementedError("bias is currently unsupported for MoE")
        self.return_bias = return_bias
        self.tensor_parallel_group = (
            tensor_model_parallel_group
            if tensor_model_parallel_group is not None
            else get_tensor_model_parallel_group()
        )
        self.expert_model_parallel_group = (
            expert_model_parallel_group
            if expert_model_parallel_group is not None
            else get_expert_model_parallel_group()
        )

        if routed_experts_mlp_config.enable_spmd_rank:
            # spmd_rank will be removed once we support ReplicaID (P87857655)
            self.spmd_rank = SPMDRank(
                world_size=parallel_state.get_world_group().size(),
                tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_group().size(),
            )

        if self.expert_model_parallel_group.size() > 1:
            logger.warning(
                f"enable_spmd_rank set to {self.routed_experts_mlp_config.enable_spmd_rank}, enable_spmd_rank must be set to True when using expert parallelism under SPMD flow"
            )

        self.mlp_op = Experts(
            num_experts=routed_experts_mlp_config.num_experts,
            hidden_size=routed_experts_mlp_config.hidden_size,
            intermediate_size=routed_experts_mlp_config.intermediate_size,
            glu=routed_experts_mlp_config.glu_mlp,
            activation_fn=ACT2FN[routed_experts_mlp_config.hidden_act],
            dtype=dtype,
            device=device,
            input_layer_init_method=routed_experts_mlp_config.input_layer_init_method,
            output_layer_init_method=routed_experts_mlp_config.output_layer_init_method,
            tensor_model_parallel_group=self.tensor_parallel_group,
            glu_activation_fn=glu_activation_fn,
        )
        self.dtype = dtype
        self.device = device

        if self.expert_model_parallel_group.size() > 1 and not self.training:
            self.spmd_rank.initialize_expert_indices(num_local_experts=self.mlp_op.gate_up_proj._n_local_experts)

    @staticmethod
    def validate_routed_experts_configs(routed_experts_mlp_config: RoutedExpertsMLPOpsConfig):
        if not (0 < routed_experts_mlp_config.top_k <= routed_experts_mlp_config.num_experts):
            raise ValueError(
                f"Invalid top_k={routed_experts_mlp_config.top_k} for num_experts={routed_experts_mlp_config.num_experts}"
            )
        if routed_experts_mlp_config.hidden_act not in ACT2FN:
            raise ValueError(
                f"Unknown activation: {routed_experts_mlp_config.hidden_act} ; Supported: {list(ACT2FN.keys())}"
            )
        if (
            routed_experts_mlp_config.capacity_factor is None
            or routed_experts_mlp_config.capacity_factor
            >= routed_experts_mlp_config.num_experts / routed_experts_mlp_config.top_k
        ):
            routed_experts_mlp_config.capacity_factor = None  # Denotes full capacity
        if routed_experts_mlp_config.normalize_top_k_affinities and routed_experts_mlp_config.top_k == 1:
            raise ValueError("top_k must be greater than 1 for normalizing top-k expert affinities")

    @staticmethod
    def get_expert_mask(expert_index, num_experts):
        """Helper function which computes top_k-hot encoded expert_mask from the given expert_index.

        Arguments:
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
        Returns:
            expert_mask: Tensor of shape (T, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
        """

        # Construct expert_mask from expert_index, using efficient version of one-hot encoding for xla device
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # (Use float dtype to perform computations in the vector engine for efficiency)
        # expert_mask: top_k-hot encoded expert assignment per token -> (T, E)
        expert_mask = torch.zeros(expert_index.shape[0], num_experts, device=expert_index.device, dtype=torch.float64)
        expert_num_idx_arr = torch.arange(num_experts, device=expert_index.device, dtype=torch.float64)
        top_k = expert_index.shape[1]
        for e in range(top_k):
            expert_mask += (expert_index[:, e].unsqueeze(1) == expert_num_idx_arr).to(torch.float64)

        return expert_mask

    @staticmethod
    def get_expert_affinities_masked(expert_affinities, expert_mask, normalize_top_k_affinities):
        """Helper function which computes the masked expert_affinities by selecting the chosen experts for each token,
        and normalizes the affinities if needed.

        Arguments:
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_mask: Tensor of shape (T, E), containing top_k-hot encoded experts for each token derived from
                         expert_index.
            normalize_top_k_affinities: Whether to normalize the affinities of the chosen experts before combining with the MLP outputs.
        Returns:
            expert_affinities_masked: Tensor of shape (T, E) containing the affinities of just the chosen experts for
                                      each token (after normalization if required).
        """

        # Apply expert_mask obtain the affinities for the chosen experts
        # expert_affinities_masked -> (T, E)
        expert_affinities_masked = expert_affinities.masked_fill(torch.eq(expert_mask, 0), 0)
        if normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            expert_affinities_masked = F.normalize(expert_affinities_masked, p=1.0, dim=1)

        return expert_affinities_masked

    def forward_all_experts(self, hidden_states, expert_affinities, expert_index, chosen_expert_indices=None):
        """Forward pass where all tokens are computed by all experts.
        This is equivalent to running forward_capacity_factor with full capacity (i.e. no token dropping), but
        by avoiding the permute/unpermute overhead.
        """

        if get_expert_model_parallel_size() > 1:
            raise NotImplementedError("Expert parallelism is not supported without capacity factor.")

        num_experts = expert_affinities.shape[1]
        if chosen_expert_indices is None:
            assert num_experts == self.routed_experts_mlp_config.num_experts
        else:
            assert num_experts == chosen_expert_indices.shape[0]

        # expert_mask: (T, E)
        expert_mask = self.get_expert_mask(expert_index, num_experts)
        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(
            expert_affinities,
            expert_mask,
            self.routed_experts_mlp_config.normalize_top_k_affinities,
        )

        # multiply expert affinity with hidden states
        if self.routed_experts_mlp_config.early_expert_affinity_modulation:
            mlp_input = torch.zeros_like(hidden_states)
            for e in range(num_experts):
                # TH * T1 -> TH
                mlp_input += hidden_states * expert_affinities_masked[:, e].unsqueeze(1)
        else:
            mlp_input = hidden_states

        # Pass all tokens through all experts
        # gate_up_proj: (1, T, H) @ (E, H, I) -> (E, T, I)
        # down_proj: (E, T, I) @ (E, I, H) -> (E, T, H)
        mlp_output = self.mlp_op(mlp_input.unsqueeze(0), expert_indices=chosen_expert_indices)

        output = torch.zeros(
            hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        # Scale by expert affinity and combine output
        if self.routed_experts_mlp_config.early_expert_affinity_modulation:
            for e in range(num_experts):
                # TH * T1 -> TH
                output += mlp_output[e] * expert_mask[:, e].unsqueeze(1)
        else:
            for e in range(num_experts):
                # TH * T1 -> TH
                output += mlp_output[e] * expert_affinities_masked[:, e].unsqueeze(1)

        return output

    def forward_capacity_factor(self, hidden_states, expert_affinities, expert_index):
        """Forward pass for performing Expert MLP computations, where each expert has a fixed 'expert capacity',
        i.e. maximum number of tokens that it can process. This is necessary for maintaining static shapes in the
        compilation graph, but may lead to dropped tokens in the computation.

        Expert capacity C is defined as:
            C = min(total_tokens, (total_tokens * top_k * capacity_factor) / num_experts)
        Note that when capacity_factor >= num_experts / top_k, C = total_tokens (i.e. each expert can hold all
        input tokens, and therefore no tokens are dropped).
        """

        total_tokens = hidden_states.shape[0]

        # compute expert capacity C = (total_tokens * top_k * Cf) / E
        expert_capacity = math.ceil(
            total_tokens
            * self.routed_experts_mlp_config.top_k
            * self.routed_experts_mlp_config.capacity_factor
            / self.routed_experts_mlp_config.num_experts
        )
        # expert_capacity can be upper bounded by total number of tokens, for the case when every token is routed to an expert
        expert_capacity = min(expert_capacity, total_tokens)

        # expert_mask: (T, E)
        expert_mask = self.get_expert_mask(expert_index, self.routed_experts_mlp_config.num_experts)

        # Compute the position of each token in experts, by a cumulative sum over the T dimension
        # position_in_expert: (T, E)
        position_in_expert = cumsum(expert_mask)

        # Update expert_mask by accounting for capacity factor (i.e. tokens exceeding capacity are dropped)
        expert_mask.masked_fill_(torch.gt(position_in_expert, expert_capacity), 0)

        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(
            expert_affinities, expert_mask, self.routed_experts_mlp_config.normalize_top_k_affinities
        )

        # Add expert offset to the position_in_expert
        # Perform operation in float64 to prevent precision issues due to auto-downcasting to bf16
        # expert_index_offsets: (E, )
        expert_index_offsets = (
            torch.arange(self.routed_experts_mlp_config.num_experts, device=hidden_states.device, dtype=torch.float64)
            * expert_capacity
        )
        # position_in_expert_with_offset: (T, E)
        position_in_expert_with_offset = position_in_expert + expert_index_offsets
        # Apply expert_mask
        position_in_expert_with_offset.masked_fill_(torch.eq(expert_mask, 0), 0)

        # Determine the index (with offset) of each token
        # total_tokens_idx: (T, 1)
        total_tokens_idx = torch.arange(total_tokens, device=hidden_states.device, dtype=torch.long).unsqueeze(1)
        # token_permutation_idx: (T, top_k)
        token_permutation_idx = position_in_expert_with_offset[total_tokens_idx, expert_index].to(dtype=torch.long)
        # token_permutation_idx is 1-indexed (contains 0 for dropped tokens)

        # token_assignments: (C*E+1, )
        # Account for the 1-indexed token_permutation_idx by adding an extra row
        token_assignments = torch.zeros(
            expert_capacity * self.routed_experts_mlp_config.num_experts + 1,
            device=hidden_states.device,
            dtype=torch.long,
        )
        # Perform a broadcasted assignment to map to token_idx
        token_assignments[token_permutation_idx] = total_tokens_idx + 1
        # Drop the first row (which was added to account for the 1-indexed token_permutation_idx)
        token_assignments = token_assignments[1:]
        # token_assignments: (E, C)
        token_assignments = token_assignments.view(self.routed_experts_mlp_config.num_experts, expert_capacity)
        # token_assignments is 1-indexed (contains 0 for dropped tokens)

        # Convert token_permutation_idx and token_assignments to be 0-indexed
        token_permutation_idx = token_permutation_idx - 1
        token_assignments = token_assignments - 1
        # They now contain '-1' for dropped tokens, enforce non-negative indices to avoid potential runtime OOB
        zero_tensor = torch.zeros(1, device=hidden_states.device, dtype=torch.long)
        token_permutation_idx = torch.max(token_permutation_idx, zero_tensor)
        token_assignments = torch.max(token_assignments, zero_tensor)
        # Indexing using these will result in the first token (index 0) being loaded in place of dropped tokens
        # However, the output from these will get masked out in the affinity scaling step

        # Permute hidden_states using token_assignments to get expert_aligned_hidden_states
        # expert_aligned_hidden_states: (E, C, H)
        expert_aligned_hidden_states = hidden_states[token_assignments, :]

        # Perform MLP operations
        # expert_aligned_output: (E, C, H)
        expert_aligned_output = self.mlp_op(expert_aligned_hidden_states)

        # convert back (E, C, H) into (C*E, H)
        permuted_output = expert_aligned_output.view(expert_capacity * self.routed_experts_mlp_config.num_experts, -1)

        # output: (T, H)
        output = torch.zeros(
            total_tokens, hidden_states.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
        )
        for k in range(self.routed_experts_mlp_config.top_k):
            # Unpermute output from the kth chosen expert for each token using token_permutation_idx
            output_k = permuted_output[token_permutation_idx[:, k]]
            expert_affinities_k = expert_affinities_masked[total_tokens_idx, expert_index[:, k].unsqueeze(1)]
            # Multiplying with the expert_affinities masks out the output of dropped tokens
            # (T, H) * (T, 1)
            output += output_k * expert_affinities_k

        return output

    def forward_selective_loading(self, hidden_states, expert_affinities, expert_index):
        """Forward pass which selectively loads only the experts chosen for each input token, during token generation."""

        T = hidden_states.shape[0]

        # chosen_expert_affinities: (T, top_k)
        chosen_expert_affinities = expert_affinities[
            torch.arange(T, device=hidden_states.device).unsqueeze(1), expert_index
        ]
        if self.routed_experts_mlp_config.normalize_top_k_affinities:
            # Normalize the affinities across the chosen experts
            chosen_expert_affinities = F.normalize(chosen_expert_affinities, p=1.0, dim=1)

        output_list = []
        for t in range(T):
            if self.routed_experts_mlp_config.early_expert_affinity_modulation:
                weighted_hidden = hidden_states[t].unsqueeze(0) * chosen_expert_affinities[t].unsqueeze(1)
                mlp_output_t = self.mlp_op(weighted_hidden.unsqueeze(1), expert_indices=expert_index[t])
                output_t = torch.sum(mlp_output_t.squeeze(1), dim=0)
            else:
                # gate_up_proj: (1, 1, H) @ (top_k, H, I) -> (top_k, 1, I)
                # down_proj: (top_k, 1, I) @ (top_k, I, H) -> (top_k, 1, H)
                mlp_output_t = self.mlp_op(hidden_states[t].unsqueeze(0).unsqueeze(1), expert_indices=expert_index[t])
                # output_t: sum((top_k, H) * (top_k, 1), dim=0) -> H
                output_t = torch.sum(mlp_output_t.squeeze(1) * chosen_expert_affinities[t].unsqueeze(1), dim=0)
            output_list.append(output_t)

        # output: (T, H)
        output = torch.stack(output_list, dim=0)

        return output

    def forward_blockwise(self, hidden_states, expert_affinities, expert_index):
        """
        Forward pass which implements the blockwise matmul approach for expert computations without
        dropping tokens.

        The tokens are assembled into fixed sized 'blocks', each of which is assigned to a single expert.
        The outputs across blocks are un-permuted and combined to obtain the output corresponding to
        each input token.

        The 'block size', i.e. the number of tokens in each block, is a hyper-parameter that must be chosen
        carefully. Larger block sizes result in better hardware utilization, but may also lead to large
        padding overheads (especially at smaller sequence lengths).
        """
        total_tokens, hidden_size = hidden_states.shape

        # num_blocks N = CEIL(((T*top_k)-(E-1))/ B) + (E-1)
        # N is the number of blocks needed in the worst case distribution of tokens to experts. Intuition below.
        # With top_k = 1, let E-1 experts be assigned 1 token each. They require 1 block each, and (E-1) total. The remaining
        # tokens are all assigned the same experts, and require CEIL((T-(E-1))/B) blocks. The formula holds for top_k > 1 also.
        if self.expert_model_parallel_group.size() > 1 and not self.training:
            assert self.mlp_op.gate_up_proj._n_local_experts == self.mlp_op.down_proj._n_local_experts
            local_experts = self.mlp_op.gate_up_proj._n_local_experts
        else:
            local_experts = self.routed_experts_mlp_config.num_experts

        num_blocks = (
            math.ceil(
                (total_tokens * self.routed_experts_mlp_config.top_k - (local_experts - 1))
                / self.blockwise_matmul_config.block_size
            )
            + local_experts
            - 1
        )
        # Handle case where T*top_k is smaller than E. We will need atmost T*top_k blocks.
        num_blocks = min(num_blocks, total_tokens * self.routed_experts_mlp_config.top_k)

        # expert_mask: (T, E). Still happens on all the experts, not just the local experts
        expert_mask = self.get_expert_mask(expert_index, self.routed_experts_mlp_config.num_experts)
        # expert_affinities_masked: (T, E)
        expert_affinities_masked = self.get_expert_affinities_masked(
            expert_affinities,
            expert_mask,
            self.routed_experts_mlp_config.normalize_top_k_affinities,
        )

        if self.expert_model_parallel_group.size() > 1 and not self.training:
            # [T, E/ep_size]
            local_expert_indices = self.spmd_rank.get_local_expert_indices()
            broadcasted_local_expert_indices = torch.broadcast_to(local_expert_indices, (total_tokens, local_experts))

            local_expert_mask = torch.gather(expert_mask, 1, broadcasted_local_expert_indices)
            local_expert_affinities_masked = torch.gather(
                expert_affinities_masked, 1, broadcasted_local_expert_indices
            )
            # TODO: make EP work with optimized_block_to_token_mapping in the future, currently not supported.
            ues_optimized_block_to_token_mapping = False
            logger.info(
                "Expert Parallel Enabled for forward_blockwise\n"
                + f"broadcasted_local_expert_indices: {broadcasted_local_expert_indices.shape}\n"
                + f"expert_mask: {expert_mask.shape}\n"
                + f"expert_affinities_masked: {expert_affinities_masked.shape}\n"
                + f"local_expert_mask: {broadcasted_local_expert_indices.shape}\n"
                + f"local_expert_affinities_masked: {local_expert_affinities_masked.shape}\n"
            )
        else:
            # [T, E]
            local_expert_mask = expert_mask
            local_expert_affinities_masked = expert_affinities_masked
            ues_optimized_block_to_token_mapping = (self.blockwise_matmul_config.optimized_block_to_token_mapping,)

        block_to_expert, token_position_to_id = self.get_blockwise_expert_and_token_mapping(
            total_tokens=total_tokens,
            num_blocks=num_blocks,
            expert_mask=local_expert_mask,
            expert_index=expert_index,
            block_size=self.blockwise_matmul_config.block_size,
            device=hidden_states.device,
            enable_spmd_rank=self.routed_experts_mlp_config.enable_spmd_rank,
            spmd_rank=self.spmd_rank if self.routed_experts_mlp_config.enable_spmd_rank else None,
            tensor_parallel_group=self.tensor_parallel_group,
            optimized_block_to_token_mapping=ues_optimized_block_to_token_mapping,
            parallelize_token_to_block_mapping=self.blockwise_matmul_config.parallelize_token_to_block_mapping,
        )

        use_blockwise_matmul_nki = can_use_blockwise_matmul_nki(
            hidden_size=hidden_size,
            intermediate_size_tp=self.mlp_op.down_proj.weight.shape[1],
            block_size=self.blockwise_matmul_config.block_size,
            glu_mlp=self.routed_experts_mlp_config.glu_mlp,
            device=hidden_states.device,
            logical_nc_config=self.blockwise_matmul_config.logical_nc_config,
            use_block_parallel=self.blockwise_matmul_config.use_block_parallel,
            use_torch_block_wise=self.blockwise_matmul_config.use_torch_block_wise,
        )
        if use_blockwise_matmul_nki:
            expert_affinities_scaling_mode = ExpertAffinityScaleMode.POST_SCALE
            if self.routed_experts_mlp_config.early_expert_affinity_modulation:
                expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

            blockwise_nki_autograd_cls = (
                self.blockwise_matmul_config.blockwise_nki_autograd_cls
                if self.blockwise_matmul_config.blockwise_nki_autograd_cls
                else BlockwiseMatmulNKIFunc
            )

            return blockwise_nki_autograd_cls.apply(
                hidden_states,
                local_expert_affinities_masked,
                self.mlp_op.gate_up_proj.weight,
                self.mlp_op.down_proj.weight,
                token_position_to_id,
                block_to_expert,
                getattr(self.mlp_op.gate_up_proj, "scale", None),
                getattr(self.mlp_op.down_proj, "scale", None),
                self.training,
                self.blockwise_matmul_config,
                self.routed_experts_mlp_config.top_k > 1,
                expert_affinities_scaling_mode,
            )
        elif self.training:
            # we split training/inference torch blockwise because training backward pass implements a simplified version of mlp_op.
            return TorchBlockwiseTraining.apply(
                hidden_states,
                expert_affinities_masked,
                token_position_to_id,
                block_to_expert,
                self.mlp_op.gate_up_proj.weight,
                self.mlp_op.down_proj.weight,
            )
        else:
            return self.torch_blockwise_matmul_inference(
                block_size=self.blockwise_matmul_config.block_size,
                num_blocks=num_blocks,
                hidden_states=hidden_states,
                expert_affinities_masked=local_expert_affinities_masked,
                token_position_to_id=token_position_to_id,
                block_to_expert=block_to_expert,
                pad_inputs_for_matmul=not self.blockwise_matmul_config.skip_dma.skip_token,
            )

    @staticmethod
    def get_blockwise_expert_and_token_mapping(
        total_tokens,
        num_blocks,
        expert_mask,
        expert_index,
        block_size,
        device,
        enable_spmd_rank,
        spmd_rank,
        tensor_parallel_group,
        optimized_block_to_token_mapping=True,
        parallelize_token_to_block_mapping=False,
    ):
        """
        Token position: position in blocks.
        E.g. given block_size=2, num_token=6. The following expert_mask
            [1, 0, 0],  # First token for expert 0
            [0, 1, 0],  # First token for expert 1
            [1, 0, 0],  # Second token for expert 0
            [0, 0, 1],  # First token for expert 2
            [1, 0, 0],  # Third token for expert 0
            [0, 1, 0],  # Second token for expert 1
        would put tokens into blocks as follows:
            Block 0 (expert 0)
                0
                2
            Block 1 (expert 0)
                4
                -1
            Block 2 (expert 1)
                1
                5
            Block 3 (expert 2)
                3
                -1
        This would result in block to expert mappping:
            0 -> 0
            1 -> 0
            2 -> 1
            3 -> 2
        and token position to id mapping:
            0 -> 0
            1 -> 2
            2 -> 4
            3 -> -1
            4 -> 1
            5 -> 5
            6 -> 3
            7 -> -1
        """

        # tokens_per_expert: (E, )
        tokens_per_expert = torch.sum(expert_mask, dim=0)
        # blocks_per_expert: (E, )
        blocks_per_expert = ((tokens_per_expert + block_size - 1) // block_size).to(dtype=torch.long)
        # block_to_expert: (N, ). Block id to expert id mapping.
        # The simplest way to do this is to use repeat_interleave after padding blocks_per_expert with unassigned blocks.
        # But this op is not lowered to xla with vector 'repeats', so we use the equivalent implementation below.
        block_ids = torch.arange(num_blocks, device=device, dtype=torch.long)  # (N, )
        cumulative_blocks_per_expert = cumsum(blocks_per_expert.unsqueeze(1)).squeeze(1)  # (E, )
        block_to_expert = torch.sum(block_ids.unsqueeze(1) >= cumulative_blocks_per_expert[:-1], dim=1).to(torch.long)

        # token_position_by_id_and_expert: (T, E)
        token_position_by_id_and_expert = cumsum(expert_mask)
        # Tokens assigned to a given expert are assembled in consecutive blocks (and will have consecutive positions)
        # The block position for a token for an expert depends on the number of blocks assigned to all previous experts.
        # Compute and add this offset for each expert
        expert_block_offsets = cumulative_blocks_per_expert * block_size
        token_position_by_id_and_expert[:, 1:] += expert_block_offsets[:-1]
        # Apply expert_mask
        token_position_by_id_and_expert = token_position_by_id_and_expert.masked_fill(torch.eq(expert_mask, 0), 0).to(
            dtype=torch.long
        )

        # Invert token_position_by_id_and_expert to obtain token_position_to_id
        # token_position_to_id: (N*B+1,)
        # Initialize with -1 indices (which will remain as the indices of padding tokens)
        token_position_to_id = -1 * torch.ones(num_blocks * block_size + 1, device=device, dtype=torch.long)

        if total_tokens % get_tensor_model_parallel_size() != 0:
            # Pad token_position_by_id_and_expert
            num_pad = (-total_tokens) % get_tensor_model_parallel_size()
            token_position_by_id_and_expert = F.pad(token_position_by_id_and_expert, (0, 0, 0, num_pad))
            tokens_idx = torch.arange(total_tokens + num_pad, device=device, dtype=torch.long)
        else:
            tokens_idx = torch.arange(total_tokens, device=device, dtype=torch.long)
        # Further reduce token_position_by_id_and_expert to only keep topk experts' token positions
        if optimized_block_to_token_mapping:
            token_position_by_id_and_expert = torch.gather(token_position_by_id_and_expert, dim=1, index=expert_index)
        if not parallelize_token_to_block_mapping:
            # token_position_to_id is a flattened array that contains the mapping of token indices for each block
            # token_position_to_id contains -1 as the index of 'padding' tokens
            token_position_to_id[token_position_by_id_and_expert] = tokens_idx.unsqueeze(1)
            token_position_to_id = token_position_to_id[1:]
            return block_to_expert, token_position_to_id
        # Distribute computation by splitting token_position_by_id_and_expert and tokens_idx across TP ranks
        # The same token_position_by_id_and_expert and tokens_idx is present at all ranks, so we can use any of MIN/MAX/AVG to as the reduce operation
        if enable_spmd_rank:
            # use rank information is available at runtime in inference
            # get tp_rank from global rank
            # note: we use `get_tensor_model_parallel_group()` here to parallelize within a single node
            #       but may get replicated in multi-node case
            tp_rank = torch.remainder(spmd_rank.get_rank(), get_tensor_model_parallel_size())
            token_position_by_id_and_expert = mappings.scatter_to_process_group_spmd(
                token_position_by_id_and_expert,
                partition_dim=0,
                rank=tp_rank,
                process_group=get_tensor_model_parallel_group(),
            )
            tokens_idx = mappings.scatter_to_process_group_spmd(
                tokens_idx,
                partition_dim=0,
                rank=tp_rank,
                process_group=get_tensor_model_parallel_group(),
            )
            # Assemble token_position_to_id using chunk of token_position_by_id_and_expert and tokens_idx at each TP rank
            # This generates small DMA transfers because of discontinuous writes, and benefits from distributing across TP ranks
            token_position_to_id[token_position_by_id_and_expert] = tokens_idx.unsqueeze(1)
            token_position_to_id = token_position_to_id[1:]
            # Accumulate results across TP ranks (use MAX to correctly account for the -1 index initialization)
            token_position_to_id = mappings._reduce(
                token_position_to_id,
                computation=xm.REDUCE_MAX,
                process_group=get_tensor_model_parallel_group(),
            )
        else:
            # To avoid rank-specific computations, we do a reduce-scatter instead of a simple split
            token_position_by_id_and_expert = mappings._reduce_scatter_along_first_dim(
                token_position_by_id_and_expert,
                computation=xm.REDUCE_MIN,
                process_group=tensor_parallel_group,
            )
            tokens_idx = mappings._reduce_scatter_along_first_dim(
                tokens_idx,
                computation=xm.REDUCE_MIN,
                process_group=tensor_parallel_group,
            )
            # Assemble token_position_to_id using chunk of token_position_by_id_and_expert and tokens_idx at each TP rank
            # This generates small DMA transfers because of discontinuous writes, and benefits from distributing across TP ranks
            token_position_to_id[token_position_by_id_and_expert] = tokens_idx.unsqueeze(1)
            token_position_to_id = token_position_to_id[1:]
            # Accumulate results across TP ranks (use MAX to correctly account for the -1 index initialization)
            token_position_to_id = mappings._reduce(
                token_position_to_id,
                computation=xm.REDUCE_MAX,
                process_group=tensor_parallel_group,
            )
        return block_to_expert, token_position_to_id

    def torch_blockwise_matmul_inference(
        self,
        block_size,
        num_blocks,
        hidden_states,
        expert_affinities_masked,
        token_position_to_id,
        block_to_expert,
        pad_inputs_for_matmul=False,
    ):
        """
        PyTorch implementation of the blockwise matmul.

        This is used when running on GPU, or when the blockwise NKI kernel is not compatible with the model
        configuration.
        """
        total_tokens, hidden_size = hidden_states.shape
        output = torch.zeros(total_tokens, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype)
        # this simulates the case (when using blockwise nki kernel) when skip_tokens in skip_dma is false,
        # where -1s in token_position_to_id needs to be converted to total_tokens, hidden_states etc needs to be padded by
        # one extra row
        if pad_inputs_for_matmul:
            output, hidden_states, token_position_to_id, expert_affinities_masked = (
                augment_inputs_for_padded_blockwise_matmul(
                    output, hidden_states, token_position_to_id, expert_affinities_masked
                )
            )
        else:
            # the extra row is used to store the output for the padded indices in the blocks which will be discarded
            # as the outputs for padding indices should not contribute to the final output
            output = torch.cat([output, torch.zeros(1, hidden_size, device=output.device, dtype=output.dtype)])
        # block_to_token_indices: (N, B)
        block_to_token_indices = token_position_to_id.view(num_blocks, block_size)
        for block_idx in range(num_blocks):
            block_token_indices = block_to_token_indices[block_idx]
            block_expert_idx = block_to_expert[block_idx]
            if self.routed_experts_mlp_config.early_expert_affinity_modulation:
                # Scale the input hidden states before MLP
                block_hidden_states = (
                    hidden_states[block_token_indices]
                    * expert_affinities_masked[block_token_indices, block_expert_idx.unsqueeze(0)].unsqueeze(1)
                ).unsqueeze(0)
                block_output = self.mlp_op(block_hidden_states, expert_indices=block_expert_idx.unsqueeze(0)).squeeze(
                    0
                )
            else:
                # block_hidden_states: (1, B, H)
                block_hidden_states = hidden_states[block_token_indices].unsqueeze(0)
                # block_mlp_output: (B, H)
                block_mlp_output = self.mlp_op(
                    block_hidden_states, expert_indices=block_expert_idx.unsqueeze(0)
                ).squeeze(0)
                # block_output: (B, H)
                # FIXME: remove unsqueeze(0) from block_expert_idx.unsqueeze(0) would OOM
                block_output = block_mlp_output * expert_affinities_masked[
                    block_token_indices, block_expert_idx.unsqueeze(0)
                ].unsqueeze(1)
            # Update the tokens computed by the block
            output[block_token_indices] += block_output

        # Drop the last row that store the output from the padded indices in the block
        output = output[:total_tokens, :]

        return output

    def forward(self, hidden_states, expert_affinities, expert_index, seq_len):
        """Forward pass of the ExpertMLPs.

        For training:
        1. If capacity_factor is None (full capacity), run forward_all_experts().
        2. Else if capacity_factor is smaller than or equal to zero, run forward_blockwise().
        3. Else run forward_capacity_factor().

        For inference:
        1. If token generation (seq len == 1):
            Run forward_selective_loading() or forward_all_experts() depending on the following logic.
            Let T be the total number of tokens. Using selective loading, T*top_k experts will be loaded.
            If (T*top_k/num_experts) is less than SELECTIVE_LOADING_THRESHOLD, then we use selective loading.
            Otherwise, we use forward_all_experts (for better performance).
        2. Else (seq len > 1):
            a. If capacity factor is not None, run forward_capacity_factor().
            b. If the number of tokens is very small such that the number of experts loaded is less
               than SELECTIVE_LOADING_THRESHOLD, use selective loading. This is useful for the speculation use case.
            c. If number of tokens * top_k is less than the block size, run forward_all_experts().
            c. Otherwise, run forward_blockwise().

        Note on the SELECTIVE_LOADING_THRESHOLD:
        This parameter determines when forward_selective_loading is used for token-gen (in favor of
        forward_all_experts), and should be a float <= 1.

        Common nomenclature:
            S: Sequence length, B: Batch size, H: Hidden Size
            T: Tokens = S * B (token dimension obtained by flattening S and B)

        Arguments:
            hidden_states: Tensor of shape (T, H).
            expert_affinities: Tensor of shape (T, E), containing the normalized affinities of each token for each expert.
            expert_index: Tensor of shape (T, top_k), containing the 'chosen' experts for each token.
            seq_len: Sequence length S. Used to infer context encoding vs token generation in inference.

        Returns:
            output: Output tensor of the same shape as hidden_states, obtained by passing each token through its assigned experts,
                    combined with the corresponding expert affinities.
        """

        if self.training:
            # Training flow
            if self.routed_experts_mlp_config.capacity_factor is None:
                return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
            elif self.routed_experts_mlp_config.capacity_factor <= 0:
                act_fn = ACT2FN[self.routed_experts_mlp_config.hidden_act]
                if act_fn != F.silu:
                    logger.info(
                        rmsg(
                            f"{act_fn=}. Blockwise training only supports SiLU activation, falling back to full capacity"
                        )
                    )
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
                if not self.routed_experts_mlp_config.glu_mlp:
                    logger.info(
                        rmsg(
                            f"{self.routed_experts_mlp_config.glu_mlp=}. Blockwise training only supports glu_mlp=True, falling back to full capacity"
                        )
                    )
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
                return self.forward_blockwise(hidden_states, expert_affinities, expert_index)
            else:
                return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
        else:
            # Inference flow
            total_tokens = hidden_states.shape[0]
            if seq_len == 1:
                # Token generation
                if get_expert_model_parallel_size() > 1:
                    raise NotImplementedError("Expert parallelism is not supported in token generation.")

                perc_experts_loaded = (
                    total_tokens * self.routed_experts_mlp_config.top_k / self.routed_experts_mlp_config.num_experts
                )
                if perc_experts_loaded >= DEFAULT_SELECTIVE_LOADING_THRESHOLD:
                    return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                else:
                    return self.forward_selective_loading(hidden_states, expert_affinities, expert_index)
            else:
                # Context Encoding / Speculative Decoding
                if self.routed_experts_mlp_config.capacity_factor is None:
                    perc_experts_loaded = (
                        total_tokens
                        * self.routed_experts_mlp_config.top_k
                        / self.routed_experts_mlp_config.num_experts
                    )
                    if perc_experts_loaded < DEFAULT_SELECTIVE_LOADING_THRESHOLD:
                        # Use selective loading for small speculation lengths
                        return self.forward_selective_loading(hidden_states, expert_affinities, expert_index)
                    elif total_tokens * self.routed_experts_mlp_config.top_k < self.blockwise_matmul_config.block_size:
                        # Use all experts for large speculation lengths, and small context encoding prompt sizes
                        # (more efficient to run all_experts instead of blockwise - equivalent in FLOPs, lower memory bandwidth usage)
                        return self.forward_all_experts(hidden_states, expert_affinities, expert_index)
                    else:
                        # Use blockwise for dropless context encoding
                        return self.forward_blockwise(hidden_states, expert_affinities, expert_index)
                else:
                    return self.forward_capacity_factor(hidden_states, expert_affinities, expert_index)
