def pytest_collection_modifyitems(items):
    """Reorder tests so @subprocess_test tests run before all others.

    Tests decorated with @subprocess_test are run in isolated subprocesses to
    prevent Neuron Runtime (NRT) state pollution.  However, they must execute
    *before* any session-scoped fixture loads a model onto NeuronCores (e.g.
    neuron_llm_config with tp_degree=2), because the parent process holding
    NeuronCores would prevent the subprocess from accessing the device.

    This hook moves all @subprocess_test items to the front of the collection.
    """
    subprocess_tests = []
    other_tests = []
    for item in items:
        if getattr(item.obj, "_subprocess_unwrapped", False):
            subprocess_tests.append(item)
        else:
            other_tests.append(item)
    items[:] = subprocess_tests + other_tests
