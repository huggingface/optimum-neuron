import functools

import pytest

from optimum.neuron.version import __sdk_version__


def skip_for_sdk(sdk_versions):
    def skip_for_sdk_decorator(test):
        @functools.wraps(test)
        def test_wrapper(*args, **kwargs):
            if __sdk_version__ in sdk_versions:
                pytest.skip(f"Fails with Neuron SDK {__sdk_version__}")
            test(*args, **kwargs)

        return test_wrapper

    return skip_for_sdk_decorator
