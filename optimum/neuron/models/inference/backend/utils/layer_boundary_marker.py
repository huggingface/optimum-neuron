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
from torch import nn
from torch_neuronx.xla_impl.base import xla_hlo_call
from torch_neuronx.xla_impl.custom_call_targets import (
    AwsNeuronModuleMarkerEndForward,
    AwsNeuronModuleMarkerStartForward,
)


# does not support nesting since we do not pass config str
class ModuleMarkerStart(torch.autograd.Function):
    @xla_hlo_call
    def forward_impl(*args):
        if len(args) == 1:
            out_ty = args[0]
        else:
            out_ty = args[0].scribe.tuple(*args)
        return out_ty.CustomCall(*args, custom_call_target=AwsNeuronModuleMarkerStartForward)

    @staticmethod
    def forward(ctx, *args):
        ctx.save_for_backward(*args)
        return ModuleMarkerStart.forward_impl(*args)

    @staticmethod
    def backward(ctx, grad_output):
        pass


# does not support nesting since we do not pass config str
class ModuleMarkerEnd(torch.autograd.Function):
    @xla_hlo_call
    def forward_impl(*args):
        if len(args) == 1:
            out_ty = args[0]
        else:
            out_ty = args[0].scribe.tuple(*args)
        return out_ty.CustomCall(*args, custom_call_target=AwsNeuronModuleMarkerEndForward)

    @staticmethod
    def forward(ctx, *args):
        ctx.save_for_backward(*args)
        return ModuleMarkerEnd.forward_impl(*args)

    @staticmethod
    def backward(ctx, grad_output):
        pass


class ModuleMarkerStartWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return ModuleMarkerStart.apply(*args)


class ModuleMarkerEndWrapper(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return ModuleMarkerEnd.apply(*args)
