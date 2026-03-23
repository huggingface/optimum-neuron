# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.

import glob
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
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


def _filtered_keys(export_keys, setup_keys):
    return sorted(set(export_keys) - set(setup_keys))


def _filtered_or_raw_keys(export_keys, setup_keys):
    filtered = _filtered_keys(export_keys, setup_keys)
    return filtered if filtered else sorted(set(export_keys))


@contextmanager
def _temporary_neuron_env():
    """Context manager for temporarily setting NEURON_COMPILE_CACHE_URL and CUSTOM_CACHE_REPO."""
    previous_env = {name: os.environ.get(name) for name in ["NEURON_COMPILE_CACHE_URL", "CUSTOM_CACHE_REPO"]}
    try:
        yield
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run_same_process_test(fn, example_inputs, workdir1, workdir2, cache_path):
    """
    Run same-process cache key determinism test.

    Args:
        fn: Function to compile
        example_inputs: Example inputs for the function
        workdir1: First compiler workdir
        workdir2: Second compiler workdir
        cache_path: Cache directory path
    Returns:
        tuple: (first_export_keys, second_export_keys, setup_keys, first_export_keys_raw, second_export_keys_raw)
    """

    # Warmup compile to capture one-time setup MODULE_* artifacts.
    def warmup_fn(x):
        return x + 1.0

    warmup_inputs = [(torch.randn(1, 1, dtype=torch.float32),)]
    build_function(
        warmup_fn,
        warmup_inputs,
        tp_degree=1,
        compiler_workdir=workdir1,
        dry_run=True,
        initialize_model_weights=False,
    )
    setup_keys = _module_keys(cache_path)
    assert len(setup_keys) > 0

    _clear_dir(cache_path)
    assert len(_module_keys(cache_path)) == 0

    build_function(
        fn,
        example_inputs,
        tp_degree=1,
        compiler_workdir=workdir1,
        dry_run=True,
        initialize_model_weights=False,
    )
    first_export_keys_raw = _module_keys(cache_path)
    first_export_keys = _filtered_or_raw_keys(first_export_keys_raw, setup_keys)
    assert len(first_export_keys) > 0

    _clear_dir(cache_path)
    assert len(_module_keys(cache_path)) == 0

    build_function(
        fn,
        example_inputs,
        tp_degree=1,
        compiler_workdir=workdir2,
        dry_run=True,
        initialize_model_weights=False,
    )
    second_export_keys_raw = _module_keys(cache_path)
    second_export_keys = _filtered_or_raw_keys(second_export_keys_raw, setup_keys)
    assert len(second_export_keys) > 0

    return first_export_keys, second_export_keys, setup_keys, first_export_keys_raw, second_export_keys_raw


@pytest.mark.xfail(strict=True, reason="Known baseline nondeterminism before canonicalization")
def test_modelbuilder_simple_graph_cache_keys_drift_same_process():
    pytest.importorskip("neuronx_distributed")
    pytest.importorskip("torch_neuronx")

    with TemporaryDirectory() as cache_path, TemporaryDirectory() as workdir1, TemporaryDirectory() as workdir2:
        with _temporary_neuron_env():
            os.environ["NEURON_COMPILE_CACHE_URL"] = cache_path
            os.environ.pop("CUSTOM_CACHE_REPO", None)

            def fn(x):
                return torch.nn.functional.gelu(x) + x

            example_inputs = [(torch.randn(4, 128, dtype=torch.float32),)]

            first_export_keys, second_export_keys, setup_keys, first_export_keys_raw, second_export_keys_raw = (
                _run_same_process_test(fn, example_inputs, workdir1, workdir2, cache_path)
            )

            assert first_export_keys == second_export_keys, (
                "Expected ModelBuilder cache module keys to be identical across two same-process exports "
                "after removing one-time setup artifacts. "
                f"setup={setup_keys}, first_raw={first_export_keys_raw}, second_raw={second_export_keys_raw}, "
                f"first={first_export_keys}, second={second_export_keys}"
            )


@pytest.mark.xfail(strict=True, reason="Known baseline nondeterminism before canonicalization")
def test_modelbuilder_simple_graph_cache_keys_different_processes():
    pytest.importorskip("neuronx_distributed")
    pytest.importorskip("torch_neuronx")

    with TemporaryDirectory() as cache_path, TemporaryDirectory() as workdir1, TemporaryDirectory() as workdir2:
        # Each export runs in a fresh Python interpreter to isolate Neuron runtime state.
        script = (
            "import os, sys, glob; from pathlib import Path; "
            "import torch; sys.path.insert(0, '{repo}'); "
            "from tests.decoder.nxd_testing import build_function; "
            "os.environ['NEURON_COMPILE_CACHE_URL'] = '{cache}'; "
            "os.environ.pop('CUSTOM_CACHE_REPO', None); "
            "def fn(x): return x + 1.0 if {warmup} else torch.nn.functional.gelu(x) + x; "
            "inputs = ([(torch.randn(1, 1),)]) if {warmup} else ([(torch.randn(4, 128),)]); "
            "build_function(fn, inputs, tp_degree=1, compiler_workdir='{workdir}', dry_run=True, initialize_model_weights=False); "
            "keys = sorted({{Path(p).name for p in glob.glob('{cache}/**/MODULE_*', recursive=True) if os.path.isdir(p)}}); "
            "print('\\n'.join(keys))"
        )

        repo_root = str(Path(__file__).parent.parent.parent)
        env = {**os.environ, "NEURON_COMPILE_CACHE_URL": cache_path, "PYTHONPATH": repo_root}
        env.pop("CUSTOM_CACHE_REPO", None)

        warmup_result = subprocess.run(
            [
                sys.executable,
                "-c",
                script.format(repo=repo_root, cache=cache_path, workdir=workdir1, warmup="True"),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert warmup_result.returncode == 0, (
            "Warmup subprocess failed. "
            f"returncode={warmup_result.returncode}, stdout={warmup_result.stdout}, stderr={warmup_result.stderr}"
        )
        setup_keys = sorted(k for k in warmup_result.stdout.strip().splitlines() if k.startswith("MODULE_"))

        _clear_dir(cache_path)
        assert len(_module_keys(cache_path)) == 0

        # First export in its own process
        result1 = subprocess.run(
            [
                sys.executable,
                "-c",
                script.format(repo=repo_root, cache=cache_path, workdir=workdir1, warmup="False"),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result1.returncode == 0, (
            "First export subprocess failed. "
            f"returncode={result1.returncode}, stdout={result1.stdout}, stderr={result1.stderr}, setup={setup_keys}"
        )
        first_export_keys_raw = sorted(k for k in result1.stdout.strip().splitlines() if k.startswith("MODULE_"))
        first_export_keys = _filtered_or_raw_keys(first_export_keys_raw, setup_keys)
        assert len(first_export_keys) > 0, (
            "First export subprocess did not produce filtered MODULE_* keys. "
            f"warmup_returncode={warmup_result.returncode}, returncode={result1.returncode}, "
            f"stdout={result1.stdout}, stderr={result1.stderr}, setup={setup_keys}, first_raw={first_export_keys_raw}"
        )

        # Clear cache between exports
        _clear_dir(cache_path)
        assert len(_module_keys(cache_path)) == 0

        # Second export in its own process
        result2 = subprocess.run(
            [
                sys.executable,
                "-c",
                script.format(repo=repo_root, cache=cache_path, workdir=workdir2, warmup="False"),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result2.returncode == 0, (
            "Second export subprocess failed. "
            f"returncode={result2.returncode}, stdout={result2.stdout}, stderr={result2.stderr}, setup={setup_keys}"
        )
        second_export_keys_raw = sorted(k for k in result2.stdout.strip().splitlines() if k.startswith("MODULE_"))
        second_export_keys = _filtered_or_raw_keys(second_export_keys_raw, setup_keys)
        assert len(second_export_keys) > 0, (
            "Second export subprocess did not produce filtered MODULE_* keys. "
            f"warmup_returncode={warmup_result.returncode}, returncode={result2.returncode}, "
            f"stdout={result2.stdout}, stderr={result2.stderr}, setup={setup_keys}, second_raw={second_export_keys_raw}"
        )

        assert first_export_keys == second_export_keys, (
            "Expected ModelBuilder cache module keys to be identical across two different-process exports "
            "after removing setup artifacts. "
            f"setup={setup_keys}, first_raw={first_export_keys_raw}, second_raw={second_export_keys_raw}, "
            f"first={first_export_keys}, second={second_export_keys}"
        )
