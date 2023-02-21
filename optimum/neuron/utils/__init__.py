from packaging import version

from .. import __version__
from .import_utils import is_neuron_available, is_neuronx_available
from .training_utils import (
    FirstAndLastDataset,
    Patcher,
    patch_forward,
    patch_model,
    patch_transformers_for_neuron_sdk,
    patched_finfo,
    prepare_environment_for_neuron,
)
