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
"""Compatibility shim for `transformers.utils.fx`, removed in transformers v5.

`neuronx_distributed/pipeline/trace.py:43` does `from transformers.utils.fx import HFTracer`
whenever transformers is installed, and subclasses it on the next line, so
`import neuronx_distributed` fails outright with transformers v5, where torch.fx support
(and the whole `transformers/utils/fx.py` module) was dropped.

Until that import is guarded on the AWS side, `install_transformers_fx_shim()` appends a
finder to `sys.meta_path` that serves a minimal stand-in module. Being appended, the finder
is only consulted once the regular ones have failed, so transformers 4.x keeps its real
module and the shim retires itself the day neuronx_distributed stops importing it.

This module lives outside the `optimum.neuron` package, and imports nothing but the standard
library, because it is installed at interpreter startup by `optimum-neuron-fx-shim.pth`:
every python process in the environment pays its import. Nothing else happens until
something actually asks for `transformers.utils.fx`, at which point the stub is built and
torch is imported. Startup cost is a few milliseconds; neither torch nor transformers is
imported.

Installing it that early is what covers the processes `neuronx_distributed` spawns to trace
a model: they import `neuronx_distributed` while unpickling their work item, before any
optimum module, and tensor parallel export dies without the shim.
"""

import functools
import importlib.abc
import importlib.util
import sys
from typing import Callable, Literal, Optional, Union


__all__ = ["install_transformers_fx_shim"]

_MODULE_NAME = "transformers.utils.fx"


def _build_hf_tracer() -> type:
    import torch

    class HFTracer(torch.fx.Tracer):
        """Stand-in for `transformers.utils.fx.HFTracer`.

        neuronx_distributed only uses it as a base class of `HFTracerWrapper(NxDTracer,
        HFTracer)`. Deriving from `torch.fx.Tracer` keeps that MRO clean and gives the
        wrapper a real `is_leaf_module`.
        """

    return HFTracer


def create_wrapper(
    function: Callable,
    op_type: Union[Literal["call_function"], Literal["call_method"], Literal["get_attr"]],
    proxy_factory_fn: Optional[Callable] = None,
) -> Callable:
    """Vendored verbatim from transformers 4.57 `transformers/utils/fx.py`.

    It only depends on torch.fx, so it needs no maintenance.
    """
    import torch

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if not torch.fx._symbolic_trace.is_fx_tracing():
            return function(*args, **kwargs)

        found_proxies = []

        def check_proxy(a):
            if isinstance(a, torch.fx.Proxy):
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


def _module_getattr(name: str):
    if name.startswith("__") and name.endswith("__"):
        # The import machinery probes for dunders such as `__path__` to decide what the module
        # is: it expects a plain AttributeError when they are absent.
        raise AttributeError(name)
    raise ImportError(
        f"`{_MODULE_NAME}.{name}` is not available: transformers v5 removed its torch.fx support, "
        "and optimum-neuron only stands in for the part of that module neuronx_distributed imports."
    )


class _ShimLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module) -> None:
        module.__doc__ = __doc__
        module.HFTracer = _build_hf_tracer()
        module.create_wrapper = create_wrapper
        module.__getattr__ = _module_getattr


class _ShimFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _MODULE_NAME:
            return None
        return importlib.util.spec_from_loader(fullname, _ShimLoader())


def install_transformers_fx_shim() -> None:
    """Serve a stub `transformers.utils.fx` when transformers does not provide one.

    The finder is appended to `sys.meta_path`, so the real module always wins when it
    exists. Calling this several times is harmless.
    """
    if not any(isinstance(finder, _ShimFinder) for finder in sys.meta_path):
        sys.meta_path.append(_ShimFinder())
