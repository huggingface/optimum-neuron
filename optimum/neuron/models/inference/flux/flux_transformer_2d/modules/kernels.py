# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# This implementation is derived from the Diffusers library.
# The original codebase has been optimized and modified to achieve optimal performance
# characteristics when executed on Amazon Neuron devices.
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

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.nccl as nccl
import numpy as np
from neuronx_distributed.parallel_layers import parallel_state
from neuronxcc.nki._private_kernels.llama3_transformer import matmul_o_proj

# Global SPMD grid variables used throughout the code.
n_prgs, prg_id = 1, 0


@nki.jit
def __init_spmd_grid_size():
    """
    Initializes the spmd global variables n_prgs, prg_id
    """
    grid_ndim = nl.program_ndim()
    assert grid_ndim == 0 or grid_ndim == 1, \
        "llama3_transfomer_fwd_<tp|rmsnorm_sp> only supports no specialization or specialization along one axis"

    global n_prgs, prg_id
    if grid_ndim != 0 and nl.num_programs(axes=0) > 1:
        n_prgs = nl.num_programs(axes=0)
        prg_id = nl.program_id(axis=0)


@nki.jit
def matmul_o_proj_kernel(self_attn_out, W_o):
    """
    This is a wrapper kernel for the LNC-sequence-split out projection kernel in the Llama 3 transformer bundle found in Neuron compiler.
    It is wrapped in this way because the specific kernel on its own does not contain an all-reduce operation for the result among tensor-parallel groups.
    More importantly, the __init_spmd_grid_size method is required in this wrapper (and thus cannot be inlined with the rest of the code) to utilize LNC sharding.
    Without this LNC sharding, the matmul will be stuffed onto one LNC and that would make it impractical to use the kernel at all.
    """
    __init_spmd_grid_size()
    batch_size, _, sequence_length = self_attn_out.shape
    _, output_dim = W_o.shape
    o_proj = nl.ndarray(
        shape=[batch_size, sequence_length, output_dim],
        dtype=self_attn_out.dtype,
        buffer=nl.shared_hbm,
    )
    o_proj_temp = nl.ndarray(
        shape=[batch_size, sequence_length, output_dim],
        dtype=self_attn_out.dtype,
        buffer=nl.shared_hbm,
    )
    replica_groups = parallel_state.get_tensor_model_parallel_replica_groups()
    i_n = nl.arange(output_dim)[None, None, :]
    i_m0 = nl.arange(sequence_length)[None, :, None]
    matmul_o_proj(self_attn_out=self_attn_out, W_o=W_o, o_proj=o_proj_temp)
    nccl.all_reduce(
        op=np.add,
        srcs=[o_proj_temp],
        dsts=[o_proj[0, i_m0, i_n]],
        replica_groups=replica_groups,
        dtype=self_attn_out.dtype,
    )
    return o_proj
