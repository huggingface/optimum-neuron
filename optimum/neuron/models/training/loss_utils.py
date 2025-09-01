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


import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import EmbeddingUtility


class _ParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group(),
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = EmbeddingUtility.range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Original implementation:
        # masked_target = target.clone() - vocab_start_index
        # masked_target[target_mask] = 0
        # New xla friendly implementation:
        is_not_ignore_index_mask = (target != ignore_index).to(vocab_parallel_logits.dtype)
        target_mask = (target >= vocab_start_index) & (target < vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target = torch.mul(masked_target, target_mask.long())

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device, dtype=torch.long)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)

        # Original implementation:
        # predicted_logits[target_mask] = 0.0
        # New xla friendly implementation:
        predicted_logits = torch.mul(predicted_logits, target_mask.float())

        # All reduce is needed to get the chunks from other devices.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all devices.
        # Original implementation:
        # exp_logits = vocab_parallel_logits
        # torch.exp(vocab_parallel_logits, out=exp_logits)
        # New xla friendly implementation:
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        # Zerooing the loss for the ignored tokens.
        loss = loss * is_not_ignore_index_mask

        # Apply the reduction, to respect the torch.nn.functional.cross_entropy_loss API
        # the reduction happens only on the non-ignored tokens.
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            num_non_ignored_tokens = is_not_ignore_index_mask.sum()
            loss = loss.sum() / num_non_ignored_tokens

        ctx.reduction = reduction
        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, is_not_ignore_index_mask)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d, is_non_ignore_index_mask = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        reduction = ctx.reduction

        # All the inputs have softmax as their gradient.
        grad_input = softmax

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device, dtype=torch.long)

        if label_smoothing > 0:
            softmax_update = 1.0 - target_mask.view(-1).float()
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d.long(), masked_target_1d.long()] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d.long(), :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= target_mask.view(-1).float()

        grad_input *= is_non_ignore_index_mask.unsqueeze(dim=-1)

        if reduction == "mean":
            num_non_ignored_tokens = is_non_ignore_index_mask.sum()
            grad_input *= grad_output / num_non_ignored_tokens
        elif reduction == "sum":
            grad_input *= grad_output
        else:
            grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None, None


# Just for testing purposes, setting that to True will feed a copy of the  input to `parallel_cross_entropy` which
# changes inputs inplace. This way the original input is not transformed and can be used in tests for comparison.
_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT: bool = False


def parallel_cross_entropy(vocab_parallel_logits, target, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    """Helper function for the cross entropy."""
    if _PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT:
        vocab_parallel_logits = vocab_parallel_logits.clone()
    return _ParallelCrossEntropy.apply(vocab_parallel_logits, target, ignore_index, reduction, label_smoothing)


@torch.fx.wrap
def fixed_cross_entropy(source, target, reduction: str = "mean", ignore_index: int = -100, **kwargs):
    tp_size = get_tensor_model_parallel_size()
    if tp_size > 1:
        if _PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT:
            source = source.clone()
        loss_function = parallel_cross_entropy
    else:
        loss_function = nn.functional.cross_entropy
    loss = loss_function(source, target, ignore_index=ignore_index, reduction=reduction)
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, reduction: str = "mean", ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, reduction=reduction, ignore_index=ignore_index, **kwargs)
    return loss
