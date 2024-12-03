#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team and Amazon Web Services, Inc. All rights reserved.
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
"""
This script formats the gsm8k dataset for use by the training script (run_clm.py) in this directory.
"""

from datasets import DatasetDict, load_dataset

def format(sample):
    sample['text'] = f"<s>[INST] {sample['question']} [/INST]\n\n{sample['answer']}</s>"
    return sample

# Downloads the gsm8k dataset directly from Hugging Face.
dataset = load_dataset("gsm8k", "main")

# We need to split the dataset into a training, and validation set.
# Note gsm8k has 'test', we rename to 'validation' for our training script.
train = dataset['train']
validation = dataset['test']

# Map the format function on all elements of the training and validation splits.
# Also removes the question and answer columns we no longer need.
train = train.map(format, remove_columns=list(train.features))
validation = validation.map(format, remove_columns=list(validation.features))

# Create a new DatasetDict with our train and validation splits.
dataset = DatasetDict({"train": train, "validation": validation})

dataset.save_to_disk('dataset_formatted')
