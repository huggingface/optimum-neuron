# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# ==============================================================================

import neuronxcc.nki.language as nl
import torch
from neuronxcc.nki import FrameworkKernel
from torch_neuronx.pyhlo.scribe import HloShape

from .compiler import DataTypeConverter


class PyTorchTracedKernel(FrameworkKernel):
    dtype_converter = DataTypeConverter()

    @staticmethod
    def get_shape(hloShape):
        return tuple(hloShape.shape_proto.dimensions)

    def translate_to_neuron_dtype(self, _dtype):
        dtype = self.dtype_converter.hlo2torch(_dtype)
        if dtype == torch.bfloat16:
            dtype = nl.bfloat16
        else:
            dtype = torch.empty(1, dtype=dtype).numpy().dtype
        return dtype

    def is_framework_tensor(self, t):
        return isinstance(t, HloShape)

    def map_framework_tensor(self, hloShape):
        shape = self.get_shape(hloShape)
        dtype = hloShape.shape_proto.element_type
        return shape, dtype


def nki_call(func, *args, **kwargs):
    """
    This function applies NKI kernel function (func) to inputs (*args) in PyHLO.

    Args:
        func: NKI kernel function
        args: inputs of func
        kwargs:
            grid (grid used in NKI kernel function)
            output_HloShapes (HloShapes of outputs of NKI kernel)


    Example:
        def mixed_pyhlo_nki(x, y):
            h = x.dtype[x.sizes].Multiply(x, x)
            o = nki_call(add_kernel, h, y, grid=32, output_HloShapes=x.dtype[x.sizes])
            return o
    """

    grid = kwargs.pop("grid", None)
    return NkiHloKernel(func, grid=grid)(*args, **kwargs)


class NkiHloKernel:
    """
    This class lowers a user defined compiler kernel to PyHLO op.

    This is the FAL binding for the NKI API for compiler to program Neuron Device directly.

    Parameters:
        func: the function of the baremetal kernel defition
        grid: launch grid configuration
        kernel_attrs: List[str], string attribute to control code injection point in compiler

    There are 2 steps to use a NKI kernel:
        1) Define NKI kernel
        2) Use NKI kernel within PyHLO by nki_call

    Example:
        # 1) Define NKI Kernel

            def add_kernel(a_input, b_input, c_output):
                # Calculate tile offsets based on current 'program'
                offset_i_x = nl.program_id(0) * 128
                offset_i_y = nl.program_id(1) * 512

                # Generate tensor indices to index tensors a and b
                ix = offset_i_x + nl.arange(128)[:, None]
                iy = offset_i_y + nl.arange(512)[None, :]

                # Load input data from external memory to on-chip memory
                # We refer to an indexed portion of a tensor as an intermediate tensor
                a_tile = nl.load(a_input[ix, iy])
                b_tile = nl.load(b_input[ix, iy])

                # compute a + b
                c_tile = a_tile + b_tile

                # store the addition results back to external memory (c_output)
                nl.store(c_output[ix, iy], value=c_tile)

        # 2) Use NKI kernel by nki_call:

            def mixed_pyhlo_nki(x, y):
                grid_x = x.sizes[0] // 128
                grid_y = x.sizes[1] // 512
                h = x.dtype[x.sizes].Multiply(x, x)
                o = nki_call(nki_add, h, y, grid=(grid_x, grid_y), output_HloShapes=[y.dtype[y.sizes]])
                return o


    """

    def __init__(self, func, grid=None, **kwargs):
        self.func = func
        self.grid = ()
        if grid is not None:
            self.set_grid(grid)
        self._kernel = PyTorchTracedKernel(func_name=func.__name__, func=self.func, grid=self.grid, **kwargs)

    def set_grid(self, grid):
        if not isinstance(grid, (tuple, list)):
            grid = [grid]
        self.grid = grid

    def __call__(self, *args, output_HloShapes=None):
        if output_HloShapes is None:
            raise ValueError("output_shape should be set in NkiHloKernel !")

        if not isinstance(output_HloShapes, (list, tuple)):
            output_HloShapes = [output_HloShapes]

        input_output_HloShapes = (*args, *output_HloShapes)
        config_str, input_names, output_names = self._kernel.dump_config(*input_output_HloShapes)

        if len(output_HloShapes) > 1:
            output_HloShapes = args[0].scribe.tuple(*output_HloShapes)
        else:
            (output_HloShapes,) = output_HloShapes

        output = output_HloShapes.CustomCall(
            *args,
            backend_config=str.encode(config_str),
            custom_call_target="AwsNeuronCustomNativeKernel",
        )

        return output
