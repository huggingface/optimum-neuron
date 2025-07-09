# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/modules/autobucketing.py
from math import log2

import torch


def generate_buckets(min_length: int, max_length: int):
    if min_length == max_length:
        return [max_length]

    min_bound = int(log2(min_length))
    max_bound = round(log2(max_length))  # we use round because it creates optimal bucket spacing

    # NOTE: because range operates on [a,b), and we rounded the log2 result
    # we won't get 2**i results close to the max_length.
    # ex. we won't see bucket spacing of [128,256,512,513] or [128,256,510,512]
    buckets = [2**i for i in range(min_bound, max_bound)] + [max_length]
    return buckets


@torch.jit.script
def generation_model_bk(
    tensors: list[torch.Tensor], buckets: torch.Tensor, padding_side: str, speculation_length: int
):
    """
    The Bucket Kernel for Token Generation Models.

    1) tensors: A list of torch tensors after running through the flattener
    2) buckets: A torch.tensor of the bucket sizes
    3) padding_side: A string specifying padding side, must be "left" or "right"
    """
    # assume tensors[1] is either pos id or attention mask (seq dim == 1 => pos id)
    item = tensors[1]
    attention_mask_is_removed = item.shape[1] == 1  # indicates item is position Id
    if attention_mask_is_removed:
        position_ids = tensors[1]
        max_position_id = (
            position_ids[:, -1] + speculation_length
            if (position_ids[:, -1] + speculation_length).all() <= buckets[-1]
            else position_ids[:, -1]
        )
        bucket_mask = (buckets <= (max_position_id).unsqueeze(1)).to(torch.int)
        bucket_idx = torch.max(torch.argmin(bucket_mask, dim=1))
    else:
        attention_mask = tensors[1]
        position_ids = tensors[2]
        max_position_id = (
            position_ids[:, -1] + speculation_length
            if (position_ids[:, -1] + speculation_length).all() <= buckets[-1]
            else position_ids[:, -1]
        )
        bucket_mask = (buckets <= (max_position_id).unsqueeze(1)).to(torch.int)
        bucket_idx = torch.max(torch.argmin(bucket_mask, dim=1))
        bucket = buckets[bucket_idx]
        # slice the attention mask based on the selected bucket size
        if padding_side == "right":
            tensors[1] = torch.ops.aten.slice(attention_mask, dim=1, start=0, end=bucket)
        else:
            tensors[1] = torch.ops.aten.slice(attention_mask, dim=1, start=buckets[-1] - bucket, end=buckets[-1])

    return tensors, bucket_idx.to(torch.int)


def get_generation_model_bk():
    return generation_model_bk


@torch.jit.script
def context_encoder_bk(tensors: list[torch.Tensor], buckets, padding_side: str, pad_token: int):
    """
    The Bucket Kernel for Context Encoding Models.

    1) tensors: A list of torch tensors after running through the flattener
    2) buckets: A torch.tensor of the bucket sizes
    3) padding_side: A string specifying padding side, must be "left" or "right"
    4) pad_token: An integer representing the pad token id. Typically this is 0.
    """
    input_ids = tensors[0]

    # -----Remarks for calculating position_idx-----
    # finds the number of non pad tokens and that is the active sequence_length
    # The resulting tensor is of shape (batch_size,)
    #
    # NOTE: We derive position_ids from input_ids because
    # position_ids is eliminated from the flattener for context encoding models.
    # ----------------------------------------------
    position_idx = (input_ids != pad_token).sum(dim=1)
    position_idx = position_idx[:, None]  # shape (batch_size, 1)
    buckets = buckets[None, :]  # shape (1, seq_len)

    # -----Remarks for choosing the bucket_idx-----
    # 1. (buckets < position_idx) produces a bucket_mask where invalid buckets are 0
    # 2. We convert the boolean tensor to int because argmin doesn't support
    # boolean tensors
    # 3. We choose the minimum valid bucket, which is the first 1 value
    # 4. From the minimum valid buckets, we choose the largest bucket, otherwise
    # we'd be truncating generated tokens from longer sequences.
    # 5. DO NOT USE argmax since we monkeypatch it,
    # causing issues with torch.jit.script
    # ---------------------------------------------
    bucket_mask = (buckets < position_idx).to(torch.int)  # shape (batch_size, seq_len)
    bucket_idx = torch.max(torch.argmin(bucket_mask, dim=1))

    # select the chosen bucket after squeezing back to original form
    bucket = buckets.squeeze(0)[bucket_idx]

    new_tensors = []

    # ---------Remarks on handling padding sides-------
    # 1. slice from the opposite side for padding
    # 2. Identify seq_id tensors by shape and don't slice it
    # -------------------------------------------------
    if padding_side == "right":
        for i, tens in enumerate(tensors):
            # identifies the seq_ids, which don't need to be sliced
            if len(tens.shape) == 1:
                new_tensors.append(tens)
            else:  # all other tensors are of shape (batch_size,seq_len) so we slice on seq_len
                new_tensors.append(torch.ops.aten.slice(tens, dim=1, start=0, end=bucket))
    else:
        max_idx = buckets[-1][-1]
        for i, tens in enumerate(tensors):
            # identifies the seq_ids, which don't need to be sliced
            if len(tens.shape) == 1:
                new_tensors.append(tens)
            else:
                new_tensors.append(torch.ops.aten.slice(tens, dim=1, start=max_idx - bucket, end=max_idx))

    return new_tensors, bucket_idx.to(torch.int)


def get_context_encoder_bk():
    return context_encoder_bk
