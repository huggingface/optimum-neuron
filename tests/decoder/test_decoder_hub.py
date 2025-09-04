# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
import re
import socket
from tempfile import TemporaryDirectory

import pytest
from huggingface_hub import HfApi, get_token
from transformers import AutoConfig, AutoModelForCausalLM

from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("from_local", [False, True], ids=["llama_from_hub", "llama_from_local"])
def test_decoder_push_to_hub(from_local):
    model_id = "llamafactory/tiny-random-Llama-3"
    export_kwargs = {"batch_size": 1, "sequence_length": 1024, "tensor_parallel_size": 2, "auto_cast_type": "bf16"}
    with TemporaryDirectory() as model_path:
        if from_local:
            hub_model = AutoModelForCausalLM.from_pretrained(model_id)
            with TemporaryDirectory() as tmpdir:
                hub_model.save_pretrained(tmpdir)
                neuron_config = NeuronModelForCausalLM.get_neuron_config(
                    model_name_or_path=tmpdir, config=hub_model.config, **export_kwargs
                )
                model = NeuronModelForCausalLM.export(tmpdir, config=hub_model.config, neuron_config=neuron_config)
                # Save must happen within the context of the tmpdir or checkpoint dir is lost
                model.save_pretrained(model_path)
        else:
            config = AutoConfig.from_pretrained(model_id)
            neuron_config = NeuronModelForCausalLM.get_neuron_config(
                model_name_or_path=model_id, config=config, **export_kwargs
            )
            model = NeuronModelForCausalLM.export(
                model_id, config=config, neuron_config=neuron_config, load_weights=False
            )
            model.save_pretrained(model_path)
        # The hub model contains the checkpoint only when the model is exported from a local path
        ignore_patterns = [] if from_local else [model.CHECKPOINT_DIR + "/*"]
        hostname = socket.gethostname()
        model_name = f"neuron-testing-{hostname}-decoder-push"
        model_name += "-from-local" if from_local else "-from-hub"
        repo_id = f"optimum-internal-testing/{model_name}"
        model.push_to_hub(model_path, repo_id, token=get_token())
        api = HfApi()
        try:
            hub_files_path = api.list_repo_files(repo_id)
            for path, _, files in os.walk(model_path):
                for name in files:
                    local_file_path = os.path.join(path, name)
                    hub_file_path = os.path.relpath(local_file_path, model_path)
                    excluded = False
                    for pattern in ignore_patterns:
                        if re.compile(pattern).match(hub_file_path) is not None:
                            excluded = True
                            break
                    assert excluded or hub_file_path in hub_files_path
        finally:
            api.delete_repo(repo_id)
