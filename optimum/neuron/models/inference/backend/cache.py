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
"""Cache decorator for NeuronX compilation based on `libneuronxla`."""

import hashlib
import json
import logging
import os
import pathlib
from contextlib import contextmanager

from libneuronxla.neuron_cc_cache import CacheUrl
from torch_neuronx.xla_impl.trace import HloArtifacts, NeffArtifacts, generate_neff, hlo_compile, setup_compiler_dirs

from ....cache.hub_cache import create_hub_compile_cache_proxy
from ....utils.patching import patch_everywhere


logger = logging.getLogger("Neuron")


# Use the same hash function as in transformers_neuronx
def get_hash_module(hlo_module, flags):
    # Hashing is pretty fast and neglegible compared to compilation time
    hash_gen = hashlib.sha256()
    text = str(hlo_module)
    if flags is not None:
        if isinstance(flags, list):
            flags = "".join(flags)
        text += flags.replace(" ", "")
    hash_gen.update(text.encode("utf-8"))
    hash = str(hash_gen.hexdigest())[:20]
    return hash


@contextmanager
def neff_cache(cache_dir: str | None = None):
    """
    Context manager to patch `torch_neuronx.xla_impl.trace.generate_neff`.

    This temporarily replaces the original function in the `torch_neuronx.xla_impl.trace` module
    to use a cache for storing and retrieving compiled NEFF files.

    Usage:

        with neff_cache(cache_dir="/path/to/cache"):
            # Your code that generates NEFF files goes here
            # The cache will be used to store and retrieve compiled NEFF files
            # The original function will be restored after exiting the context

    This function uses the `libneuronxla` library to create a compile cache.
    Each entry in the cache is identified by a hash of the HLO module and its compilation flags.

    Args:
        cache_dir (`str`, *optional*):
            Directory to store the cache. If not provided, a default directory will be used.
    """

    def generate_neff_with_cache(
        hlo_artifacts: HloArtifacts,
        compiler_workdir: str | pathlib.Path | None = None,
        compiler_args: Union[list[str | None, str]] = None,
        inline_weights_to_neff: bool = True,
    ):
        """
        Generate a NEFF file from the HLO artifacts using the specified compiler arguments.

        Unlike the original implementation, this function uses a cache to store and retrieve compiled NEFF files.
        If the weights were not in the optimal layout, the compiler also produces a wrapped neff HLO stub that needs
        to be cached as well.

        Args:
            hlo_artifacts (`HloArtifacts`):
                HLO artifacts containing the HLO module and constant parameter tensors.
            compiler_workdir (`str`, *optional*):
                Directory to store the compiler workdir. If not provided, a default directory will be used.
            compiler_args (`list[str]` or `str`, *optional*):
                Compiler arguments to be used for compilation. If not provided, a default set of arguments will be used.
            inline_weights_to_neff (`bool`, *optional*):
                Whether to inline weights to NEFF. Defaults to `True`.
        Returns:
            `NeffArtifacts`:
                NEFF artifacts containing the path to the compiled NEFF file.
        """
        if inline_weights_to_neff:
            # We don't cache compilation artifacts containing weights
            return generate_neff(
                hlo_artifacts,
                compiler_workdir=compiler_workdir,
                compiler_args=compiler_args,
                inline_weights_to_neff=inline_weights_to_neff,
            )

        # Generate the HLO and other artifacts required for compilation
        compiler_target = setup_compiler_dirs(
            hlo_artifacts.hlo_module,
            compiler_workdir,
            hlo_artifacts.constant_parameter_tensors,
            inline_weights_to_neff,
        )

        # Create a hub compile cache proxying the libneuronxla cache
        # It will fetch contents from the hub if they are not found in the local cache
        cache_url = CacheUrl.get_cache_url(cache_dir=cache_dir)
        compile_cache = create_hub_compile_cache_proxy(cache_url)

        # The cache key is a hash of the HLO module and the compiler arguments
        cache_key = get_hash_module(hlo_artifacts.hlo_module, compiler_args)

        # Look in the cache
        compile_flags_str = json.dumps(compiler_args)
        entry = compile_cache.lookup(cache_key, compiler_args)

        # The result of the compilation that we need to fetch or produce in the compiler
        # working directory is composed of the NEFF file and an optional wrapped neff HLO stub
        neff_filename = os.path.join(compiler_workdir, "graph.neff")
        wrapped_neff_filename = os.path.join(compiler_workdir, "wrapped_neff.hlo")

        with entry:
            if entry.exists:
                # There is an entry in the cache, download it at the expected location
                entry.download_neff(neff_filename)
                # If the weights were not in the optimal layout, there might also be a wrapped neff HLO stub
                entry.download_wrapped_neff(wrapped_neff_filename)
                logger.info(f"Using a cached neff at {entry.neff_path}")
                return NeffArtifacts(neff_filename)

        # This graph doesn't have a NEFF in the cache yet, and we're holding the lock for it
        # First make sure the inputs are in the cache
        entry.upload_inputs(compiler_target, compile_flags_str)

        # Now compile the graph
        compiled_neff_filename = hlo_compile(
            compiler_target, compiler_workdir=compiler_workdir, compiler_args=compiler_args
        )
        if compiled_neff_filename != neff_filename:
            # The compiled NEFF file is not at the expected location, which reveals that the hlo_compile implementation has evolved
            raise ValueError(
                "Incompatible torch_neuronx.xla_impl.trace.hlo_compile implementation. Did you update the library ?"
            )

        # Store the generated artifacts in the cache
        logger.info(f"Caching neff at {entry.neff_path}")
        entry.upload_neff(neff_filename)
        if os.path.exists(wrapped_neff_filename):
            logger.info(f"Caching wrapped neff HLO stub at {entry.neff_path}")
            entry.upload_wrapped_neff(wrapped_neff_filename)

        return NeffArtifacts(neff_filename)

    try:
        patch_everywhere("generate_neff", generate_neff_with_cache, "torch_neuronx.xla_impl.trace")
        yield
    finally:
        patch_everywhere("generate_neff", generate_neff, "torch_neuronx.xla_impl.trace")
