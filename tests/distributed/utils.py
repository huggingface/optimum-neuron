# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities for tests distributed."""

from typing import Optional

import torch


def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    msg: Optional[str] = None,
):
    assert a.dtype is b.dtype, f"Expected tensors to have the same dtype, but got {a.dtype} and {b.dtype}"

    dtype = a.dtype
    if atol is None:
        atol = torch.finfo(dtype).resolution
    # Please refer to that discussion for default rtol values based on the float type:
    # https://scicomp.stackexchange.com/questions/43111/float-equality-tolerance-for-single-and-half-precision
    if rtol is None:
        rtol = {torch.float32: 1e-5, torch.float16: 1e-3, torch.bfloat16: 1e-1}[dtype]
    torch.testing.assert_close(
        a,
        b,
        atol=atol,
        rtol=rtol,
        msg=msg,
    )
