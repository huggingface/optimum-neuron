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
def _cache_key_canonicalization_context():
    from optimum.neuron.cache.canonicalization import patch_cache_key_canonicalization

    restore = patch_cache_key_canonicalization()
    try:
        yield
    finally:
        restore()


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


def _run_same_process_test(fn, example_inputs, workdir1, workdir2, cache_path, canonicalize=False):
    """
    Run same-process cache key determinism test.

    Args:
        fn: Function to compile
        example_inputs: Example inputs for the function
        workdir1: First compiler workdir
        workdir2: Second compiler workdir
        cache_path: Cache directory path
        canonicalize: Whether to use cache key canonicalization

    Returns:
        tuple: (first_export_keys, second_export_keys, setup_keys, first_export_keys_raw, second_export_keys_raw)
    """
    if canonicalize:
        context = _cache_key_canonicalization_context()
    else:

        @contextmanager
        def noop_context():
            yield

        context = noop_context()

    with context:
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


def _generate_subprocess_script(canonicalize: bool) -> str:
    """Generate subprocess script template for cache key export tests."""
    canonicalize_line = (
        (
            "from optimum.neuron.cache.canonicalization import patch_cache_key_canonicalization; "
            "patch_cache_key_canonicalization() if {canonicalize} else None; "
        )
        if canonicalize
        else ""
    )

    return (
        "import os, sys, glob; from pathlib import Path; "
        "import torch; sys.path.insert(0, '{repo}'); "
        "from tests.decoder.nxd_testing import build_function; "
        f"{canonicalize_line}"
        "os.environ['NEURON_COMPILE_CACHE_URL'] = '{cache}'; "
        "os.environ.pop('CUSTOM_CACHE_REPO', None); "
        "def fn(x): return x + 1.0 if {warmup} else torch.nn.functional.gelu(x) + x; "
        "inputs = ([(torch.randn(1, 1),)]) if {warmup} else ([(torch.randn(4, 128),)]); "
        "build_function(fn, inputs, tp_degree=1, compiler_workdir='{workdir}', dry_run=True, initialize_model_weights=False); "
        "keys = sorted({{Path(p).name for p in glob.glob('{cache}/**/MODULE_*', recursive=True) if os.path.isdir(p)}}); "
        "print('\\n'.join(keys))"
    )


def _run_subprocess_export(
    script_template: str, params: dict, env: dict, operation_name: str, setup_keys: list = None
) -> tuple:
    """
    Run cache key export in subprocess and extract MODULE_* keys.

    Args:
        script_template: String template for the subprocess script
        params: Dict of parameters to format into the template
        env: Environment dict for subprocess
        operation_name: Name of operation (for error messages)
        setup_keys: List of setup keys to filter out (optional)

    Returns:
        tuple: (filtered_keys, raw_keys)
    """
    result = subprocess.run(
        [sys.executable, "-c", script_template.format(**params)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"{operation_name} subprocess failed. "
        f"returncode={result.returncode}, stdout={result.stdout}, stderr={result.stderr}"
    )
    keys_raw = sorted(k for k in result.stdout.strip().splitlines() if k.startswith("MODULE_"))
    if setup_keys is not None:
        keys = _filtered_or_raw_keys(keys_raw, setup_keys)
    else:
        keys = keys_raw
    return keys, keys_raw


def _run_different_process_test(workdir1, workdir2, cache_path, canonicalize=False):
    """
    Run different-process cache key determinism test.

    Args:
        workdir1: First compiler workdir
        workdir2: Second compiler workdir
        cache_path: Cache directory path
        canonicalize: Whether to use cache key canonicalization

    Returns:
        tuple: (first_export_keys, second_export_keys, setup_keys, first_export_keys_raw, second_export_keys_raw)
    """
    script_template = _generate_subprocess_script(canonicalize)
    repo_root = str(Path(__file__).parent.parent.parent)
    env = {**os.environ, "NEURON_COMPILE_CACHE_URL": cache_path, "PYTHONPATH": repo_root}
    env.pop("CUSTOM_CACHE_REPO", None)

    # Warmup
    warmup_keys, _ = _run_subprocess_export(
        script_template,
        {
            "repo": repo_root,
            "cache": cache_path,
            "workdir": workdir1,
            "warmup": "True",
            "canonicalize": "True" if canonicalize else "False",
        },
        env,
        "Warmup",
    )
    setup_keys = warmup_keys

    _clear_dir(cache_path)
    assert len(_module_keys(cache_path)) == 0

    # First export
    first_export_keys, first_export_keys_raw = _run_subprocess_export(
        script_template,
        {
            "repo": repo_root,
            "cache": cache_path,
            "workdir": workdir1,
            "warmup": "False",
            "canonicalize": "True" if canonicalize else "False",
        },
        env,
        "First export",
        setup_keys,
    )
    assert len(first_export_keys) > 0, (
        f"First export subprocess did not produce filtered MODULE_* keys. "
        f"setup={setup_keys}, first_raw={first_export_keys_raw}"
    )

    _clear_dir(cache_path)
    assert len(_module_keys(cache_path)) == 0

    # Second export
    second_export_keys, second_export_keys_raw = _run_subprocess_export(
        script_template,
        {
            "repo": repo_root,
            "cache": cache_path,
            "workdir": workdir2,
            "warmup": "False",
            "canonicalize": "True" if canonicalize else "False",
        },
        env,
        "Second export",
        setup_keys,
    )
    assert len(second_export_keys) > 0, (
        f"Second export subprocess did not produce filtered MODULE_* keys. "
        f"setup={setup_keys}, second_raw={second_export_keys_raw}"
    )

    return first_export_keys, second_export_keys, setup_keys, first_export_keys_raw, second_export_keys_raw


@pytest.mark.parametrize(
    "use_canonicalization",
    [
        pytest.param(
            False, marks=pytest.mark.xfail(strict=True, reason="Known baseline nondeterminism before canonicalization")
        ),
        True,
    ],
)
def test_modelbuilder_simple_graph_cache_keys_same_process(use_canonicalization):
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
                _run_same_process_test(
                    fn, example_inputs, workdir1, workdir2, cache_path, canonicalize=use_canonicalization
                )
            )

            assert first_export_keys == second_export_keys, (
                f"Expected ModelBuilder cache module keys to be identical across two same-process exports. "
                f"setup={setup_keys}, first_raw={first_export_keys_raw}, second_raw={second_export_keys_raw}, "
                f"first={first_export_keys}, second={second_export_keys}"
            )


@pytest.mark.parametrize(
    "use_canonicalization",
    [
        pytest.param(
            False, marks=pytest.mark.xfail(strict=True, reason="Known baseline nondeterminism before canonicalization")
        ),
        True,
    ],
)
def test_modelbuilder_simple_graph_cache_keys_different_processes(use_canonicalization):
    pytest.importorskip("neuronx_distributed")
    pytest.importorskip("torch_neuronx")

    with TemporaryDirectory() as cache_path, TemporaryDirectory() as workdir1, TemporaryDirectory() as workdir2:
        first_export_keys, second_export_keys, setup_keys, first_export_keys_raw, second_export_keys_raw = (
            _run_different_process_test(workdir1, workdir2, cache_path, canonicalize=use_canonicalization)
        )

        assert first_export_keys == second_export_keys, (
            f"Expected ModelBuilder cache module keys to be identical across two different-process exports. "
            f"setup={setup_keys}, first_raw={first_export_keys_raw}, second_raw={second_export_keys_raw}, "
            f"first={first_export_keys}, second={second_export_keys}"
        )
