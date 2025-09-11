# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import shutil

from transformers import AutoTokenizer

from optimum.neuron import (
    NeuronModelForSequenceClassification,
)
from optimum.neuron.cache import get_hub_cached_entries, synchronize_hub_cache
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx

from ..cache_utils import (
    assert_local_and_hub_cache_sync,
    check_traced_cache_entry,
    get_local_cached_files,
    local_cache_size,
)


def export_encoder_model(model_id):
    batch_size = 1
    sequence_length = 64
    return NeuronModelForSequenceClassification.from_pretrained(
        model_id,
        export=True,
        dynamic_batch_size=False,
        batch_size=batch_size,
        sequence_length=sequence_length,
        inline_weights_to_neff=False,
    )


def check_encoder_inference(model, tokenizer):
    text = ["This is a sample output"]
    tokens = tokenizer(text, return_tensors="pt")
    outputs = model(**tokens)
    assert "logits" in outputs


@is_inferentia_test
@requires_neuronx
def test_encoder_cache(cache_repos):
    cache_path, cache_repo_id = cache_repos
    model_id = "hf-internal-testing/tiny-random-BertModel"
    # Export the model a first time to populate the local cache
    model = export_encoder_model(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    check_encoder_inference(model, tokenizer)
    # check registry
    check_traced_cache_entry(cache_path)
    # Synchronize the hub cache with the local cache
    synchronize_hub_cache(cache_repo_id=cache_repo_id)
    assert_local_and_hub_cache_sync(cache_path, cache_repo_id)
    # Verify we are able to fetch the cached entry for the model
    model_entries = get_hub_cached_entries(model_id, task="text-classification", cache_repo_id=cache_repo_id)
    assert len(model_entries) == 1
    # Clear the local cache
    for root, dirs, files in os.walk(cache_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    assert local_cache_size(cache_path) == 0
    # Export the model again: the compilation artifacts should be fetched from the Hub
    model = export_encoder_model(model_id)
    check_encoder_inference(model, tokenizer)
    # Verify the local cache directory has not been populated
    assert len(get_local_cached_files(cache_path, ".neuron")) == 0
