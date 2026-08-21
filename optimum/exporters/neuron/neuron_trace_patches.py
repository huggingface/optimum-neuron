# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Compatibility patches that make certain models traceable by `torch_neuronx.trace`.

Background
----------
`torch_neuronx.trace` lowers a model by running it once with tensors on an XLA
device. An ``aten`` op that has no XLA lowering falls back to CPU through
``at::native::cpu_fallback``, which calls ``XLANativeFunctions::_to_cpu()`` on the
op's operands.

That transfer only succeeds if the operand has already been materialized into a
real PjRt buffer:

* operand derived from a **model input** -> a buffer exists -> the fallback works.
* operand derived from a **parameter or buffer** -> torch-xla still holds an
  un-materialized placeholder whose buffer is ``nullptr``. The transfer then trips
  ``Check failed: pjrt_data->buffer != nullptr`` and the process dies, either with
  ``SIGABRT`` (the ``LOG(FATAL)`` path) or with a bare ``SIGSEGV``. Because the
  crash happens in C++, it cannot be caught from Python.

So a model breaks only when it applies an unlowered op to a *weight* rather than
to an activation. This affects, on Neuron SDK >= 2.27:

======================  ==============================  ===========================
Model                   Unlowered op                    Operand
======================  ==============================  ===========================
``wav2vec2``            ``aten::_weight_norm_interface``  ``weight_norm`` parametrization
``hubert``              ``aten::_weight_norm_interface``  ``weight_norm`` parametrization
``yolos``               ``aten::upsample_bicubic2d``      ``position_embeddings``
``convbert``            ``aten::im2col``                  conv-weight-derived tensor
======================  ==============================  ===========================

The patches below rewrite each of those call sites using ops that *do* have XLA
lowerings, so no CPU fallback is attempted. Every patch is mathematically
equivalent to the code it replaces; none of them changes model outputs.

Upstream tracking: https://github.com/aws-neuron/aws-neuron-sdk/issues/1265
"""

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


def fold_weight_norm_parametrizations(model: "torch.nn.Module") -> list[str]:
    """Materialize ``weight_norm`` parametrizations so tracing does not hit a CPU fallback.

    ``torch.nn.utils.weight_norm`` recomputes ``w = g * v / ||v||`` on every forward
    through ``aten::_weight_norm_interface``, which has no XLA lowering. ``g`` and
    ``v`` are parameters, so the CPU fallback receives un-materialized placeholders
    and the tracing process is killed.

    Folding the parametrization evaluates that expression once, eagerly, and stores
    the result as a plain weight. The module stays numerically identical while
    containing only lowerable ops.

    This is applied to every model, not just the ones listed in the module
    docstring: any architecture using weight normalization (speech encoders,
    vocoders, some GANs) is affected in the same way, and folding is a no-op for
    models that do not use it.

    Args:
        model: the model about to be traced. Modified in place.

    Returns:
        The names of the parametrizations that were folded.
    """
    folded = []
    for name, module in model.named_modules():
        parametrizations = getattr(module, "parametrizations", None)
        if parametrizations is not None:
            for param_name in list(parametrizations.keys()):
                torch.nn.utils.parametrize.remove_parametrizations(module, param_name, leave_parametrized=True)
                folded.append(f"{name}.{param_name}" if name else param_name)
        elif hasattr(module, "weight_g") and hasattr(module, "weight_v"):
            # weight_norm implementation used before torch 1.12
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:
                continue
            folded.append(f"{name}.weight" if name else "weight")

    if folded:
        logger.info("Folded %d weight_norm parametrization(s) for Neuron tracing: %s", len(folded), folded)
    return folded


def unfold_via_slices(
    inputs: "torch.Tensor",
    kernel_size: list[int],
    dilation: int = 1,
    padding: list[int] | int = 0,
    stride: int = 1,
) -> "torch.Tensor":
    """Replacement for ``nn.functional.unfold`` covering ConvBert's usage.

    ``nn.functional.unfold`` lowers to ``aten::im2col``, which has no XLA lowering.
    ConvBert calls it with a ``[K, 1]`` kernel over a ``[B, C, L, 1]`` tensor, unit
    stride and dilation, and padding on the sequence dimension only. That case is
    expressible with pad / slice / stack / reshape, all of which lower cleanly.

    The output matches ``nn.functional.unfold`` exactly, including its
    channel-major, kernel-minor column ordering.

    Args:
        inputs: tensor shaped ``[B, C, L, 1]``.
        kernel_size: ``[K, 1]``.
        dilation: must be 1.
        padding: ``[P, 0]`` or an int applied to the sequence dimension.
        stride: must be 1.

    Returns:
        Tensor shaped ``[B, C * K, L]``.

    Raises:
        ValueError: if called with arguments outside the supported case, so that an
            unsupported call is reported instead of silently computing something else.
    """
    kernel_height, kernel_width = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
    pad_height, pad_width = padding if isinstance(padding, (list, tuple)) else (padding, padding)

    if kernel_width != 1 or pad_width != 0 or dilation != 1 or stride != 1 or inputs.shape[-1] != 1:
        raise ValueError(
            "unfold_via_slices only supports a [K, 1] kernel over a [B, C, L, 1] tensor with "
            f"dilation=1 and stride=1, got kernel_size={kernel_size}, dilation={dilation}, "
            f"padding={padding}, stride={stride}, input shape={tuple(inputs.shape)}."
        )

    batch_size, channels, length, _ = inputs.shape
    padded = nn.functional.pad(inputs, (0, 0, pad_height, pad_height)).squeeze(-1)
    columns = torch.stack([padded[:, :, offset : offset + length] for offset in range(kernel_height)], dim=2)
    return columns.reshape(batch_size, channels * kernel_height, length)


def _patch_convbert() -> bool:
    """Route ConvBert's ``unfold`` call through :func:`unfold_via_slices`."""
    try:
        from transformers.models.convbert import modeling_convbert
    except ImportError:
        return False

    if getattr(modeling_convbert, "_neuron_unfold_patched", False):
        return True

    original_nn = modeling_convbert.nn

    class _FunctionalProxy:
        """Forwards to ``nn.functional`` but substitutes ``unfold``."""

        def __init__(self, functional):
            self._functional = functional

        def unfold(self, *args, **kwargs):
            return unfold_via_slices(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._functional, name)

    class _NNProxy:
        """Forwards to ``torch.nn`` but exposes the patched ``functional``."""

        def __init__(self, module):
            self._module = module
            self.functional = _FunctionalProxy(module.functional)

        def __getattr__(self, name):
            return getattr(self._module, name)

    modeling_convbert.nn = _NNProxy(original_nn)
    modeling_convbert._neuron_unfold_patched = True
    logger.info("Patched ConvBert to use a lowerable unfold implementation for Neuron tracing.")
    return True


def _patch_yolos(model: "torch.nn.Module") -> bool:
    """Precompute YOLOS position-embedding interpolation.

    YOLOS interpolates ``position_embeddings`` with ``mode="bicubic"``, and
    ``aten::upsample_bicubic2d`` has no XLA lowering. The result depends only on
    that parameter and on ``img_size``, which is fixed for a traced model, so it is
    a constant with respect to the traced graph. Evaluating it eagerly and caching
    it keeps the value identical while removing the op from the graph.

    The parameter values are snapshotted to CPU **here**, before tracing starts,
    and the patched forward reads only from that snapshot. Copying the live tensor
    during tracing instead would force a synchronization on an XLA tensor whose
    computation is still in flight, which fails with
    ``Check failed: handle->HasValue()``.
    """
    try:
        from transformers.models.yolos import modeling_yolos
    except ImportError:
        return False

    interpolation_classes = tuple(
        cls
        for cls in (
            getattr(modeling_yolos, "InterpolateInitialPositionEmbeddings", None),
            getattr(modeling_yolos, "InterpolateMidPositionEmbeddings", None),
        )
        if cls is not None
    )
    if not interpolation_classes:
        return False

    # Snapshot the position embeddings now, while the model is still on CPU, keyed
    # by shape. Reading the live tensor inside the patched forward instead would
    # synchronize an XLA tensor whose computation is still in flight, which fails
    # with `Check failed: handle->HasValue()`. Shape is a stable key here: the two
    # interpolation modules consume differently shaped embeddings (3D for the
    # initial one, 4D for the mid ones), and each is a single named parameter.
    snapshots = {}
    for name, param in model.named_parameters():
        if "position_embeddings" in name:
            snapshots[tuple(param.shape)] = param.detach().clone()

    if not snapshots:
        return False

    patched_any = False
    for module in model.modules():
        if not isinstance(module, interpolation_classes) or getattr(module, "_neuron_patched", False):
            continue

        def make_forward(interpolation_module):
            original_forward = type(interpolation_module).forward
            cache = {}

            def forward(pos_embed, img_size=(800, 1344)):
                key = (tuple(pos_embed.shape), tuple(img_size))
                if key not in cache:
                    source = snapshots.get(tuple(pos_embed.shape))
                    if source is None:
                        # Not a snapshotted embedding, e.g. plain CPU execution.
                        source = pos_embed.detach().cpu()
                    with torch.no_grad():
                        cache[key] = original_forward(interpolation_module, source, img_size).detach()
                return cache[key].to(device=pos_embed.device, dtype=pos_embed.dtype)

            return forward

        module.forward = make_forward(module)
        module._neuron_patched = True
        patched_any = True

    if patched_any:
        logger.info("Patched YOLOS to precompute position-embedding interpolation for Neuron tracing.")
    return patched_any


# Model types needing a patch beyond the always-applied weight_norm folding.
# Each callable takes the model and returns whether it patched anything.
_MODEL_TYPE_PATCHES = {
    "convbert": lambda model: _patch_convbert(),
    "yolos": _patch_yolos,
}


def patch_model_for_neuron_tracing(model: "torch.nn.Module") -> None:
    """Apply the compatibility patches a model needs before `torch_neuronx.trace`.

    Without this, ``wav2vec2``, ``hubert``, ``yolos`` and ``convbert`` terminate the
    tracing process with ``SIGABRT`` or ``SIGSEGV`` on Neuron SDK >= 2.27. See the
    module docstring for the mechanism.

    All patches preserve model outputs exactly, and are safe to apply to models
    that do not need them.

    Args:
        model: the model about to be traced. Modified in place.
    """
    fold_weight_norm_parametrizations(model)

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    patch = _MODEL_TYPE_PATCHES.get(model_type)
    if patch is not None:
        patch(model)
