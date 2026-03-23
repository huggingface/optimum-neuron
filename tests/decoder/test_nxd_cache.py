# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.

import glob
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from tests.decoder.nxd_testing import build_function


def _module_keys(cache_path: str):
    module_dirs = glob.glob(f"{cache_path}/**/MODULE_*", recursive=True)
    return sorted({Path(p).name for p in module_dirs if os.path.isdir(p)})


def _clear_dir(path: str):
    for root, dirs, files in os.walk(path):
        for filename in files:
            os.unlink(os.path.join(root, filename))
        for dirname in dirs:
            shutil.rmtree(os.path.join(root, dirname))


def test_modelbuilder_simple_graph_cache_keys_drift_same_process():
    pytest.importorskip("neuronx_distributed")
    pytest.importorskip("torch_neuronx")

    previous_env = {name: os.environ.get(name) for name in ["NEURON_COMPILE_CACHE_URL", "CUSTOM_CACHE_REPO"]}

    with TemporaryDirectory() as cache_path, TemporaryDirectory() as workdir1, TemporaryDirectory() as workdir2:
        os.environ["NEURON_COMPILE_CACHE_URL"] = cache_path
        os.environ.pop("CUSTOM_CACHE_REPO", None)

        try:
            # Keep the graph simple while still going through ModelBuilder trace+compile.
            fn = lambda x: torch.nn.functional.gelu(x) + x
            example_inputs = [(torch.randn(4, 128, dtype=torch.float32),)]

            build_function(fn, example_inputs, tp_degree=1, compiler_workdir=workdir1)
            first_export_keys = _module_keys(cache_path)
            assert len(first_export_keys) > 0

            _clear_dir(cache_path)
            assert len(_module_keys(cache_path)) == 0

            build_function(fn, example_inputs, tp_degree=1, compiler_workdir=workdir2)
            second_export_keys = _module_keys(cache_path)
            assert len(second_export_keys) > 0

            assert first_export_keys == second_export_keys, (
                "Expected ModelBuilder cache module keys to be identical across two same-process exports. "
                f"first={first_export_keys}, second={second_export_keys}"
            )
        finally:
            for name, value in previous_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
