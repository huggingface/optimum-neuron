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
"""Access to the `transformers.utils.fx` shim from inside the package.

The shim itself lives in the top level `_optimum_neuron_fx_shim` module, which is installed
at interpreter startup by `optimum-neuron-fx-shim.pth`: it must not import this package, so
that the processes neuronx_distributed spawns do not pay for it. Installing it again from
here is a no-op, and covers running from a source tree where the startup hook is absent.
"""

from _optimum_neuron_fx_shim import install_transformers_fx_shim


__all__ = ["install_transformers_fx_shim"]
