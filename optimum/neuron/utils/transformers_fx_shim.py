# coding=utf-8
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
"""Compatibility shim for `transformers.utils.fx`, removed in transformers v5.

`neuronx_distributed/pipeline/trace.py:43` unconditionally does
`from transformers.utils.fx import HFTracer` whenever transformers is installed, so
`import neuronx_distributed` fails outright with transformers v5, where torch.fx support
(and the whole `transformers/utils/fx.py` module) was dropped.

Until that import is guarded on the AWS side, `install_transformers_fx_shim()` registers a
minimal stand-in module. It is a no-op when the real module is importable, so it does
nothing on transformers 4.x and stops doing anything once neuronx_distributed is fixed.

Known gap: a script that does `import neuronx_distributed` *before* `import optimum.neuron`
is not covered, since the shim is installed by `optimum.neuron.__init__`.
"""

import functools
import importlib.util
import sys
from types import ModuleType
from typing import Callable, Literal, Optional, Union


__all__ = ["install_transformers_fx_shim"]


class HFTracer:
    """Stand-in for `transformers.utils.fx.HFTracer`.

    neuronx_distributed only uses it as a base class of `HFTracerWrapper(NxDTracer,
    HFTracer)`, whose MRO then resolves to `torch.fx.Tracer` through `NxDTracer`.
    """


def create_wrapper(
    function: Callable,
    op_type: Union[Literal["call_function"], Literal["call_method"], Literal["get_attr"]],
    proxy_factory_fn: Optional[Callable] = None,
) -> Callable:
    """Vendored verbatim from transformers 4.57 `transformers/utils/fx.py`.

    It only depends on torch.fx, so it needs no maintenance.
    """
    import torch
    from torch.fx import Proxy
    from torch.fx._symbolic_trace import is_fx_tracing

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if not is_fx_tracing():
            return function(*args, **kwargs)

        found_proxies = []

        def check_proxy(a):
            if isinstance(a, Proxy):
                found_proxies.append(a)

        torch.fx.node.map_aggregate(args, check_proxy)
        torch.fx.node.map_aggregate(kwargs, check_proxy)

        if len(found_proxies) > 0:
            tracer = found_proxies[0].tracer
            if op_type == "call_function":
                target = function
            elif op_type == "call_method" or op_type == "get_attr":
                target = function.__name__
            else:
                raise ValueError(f"op_type {op_type} not supported.")
            return tracer.create_proxy(op_type, target, args, kwargs, proxy_factory_fn=proxy_factory_fn)
        else:
            return function(*args, **kwargs)

    return wrapper


def install_transformers_fx_shim() -> None:
    """Register a stub `transformers.utils.fx` module when transformers does not provide one."""
    if "transformers.utils.fx" in sys.modules:
        return
    if importlib.util.find_spec("transformers.utils.fx") is not None:
        return

    module = ModuleType("transformers.utils.fx")
    module.__doc__ = __doc__
    module.HFTracer = HFTracer
    module.create_wrapper = create_wrapper
    sys.modules["transformers.utils.fx"] = module

    import transformers.utils

    transformers.utils.fx = module
