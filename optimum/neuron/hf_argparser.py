# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Customizes the HfArgumentParser to add checks for AWS Neuron instances."""

from transformers import HfArgumentParser

from .utils.argument_utils import validate_arg


class NeuronHfArgumentParser(HfArgumentParser):
    def validate_args(self, args):
        validate_arg(
            args,
            "pad_to_max_length",
            (
                "pad_to_max_length=False can lead to very poor performance because it can trigger a lot of recompilation "
                "due to variable sequence length."
            ),
            expected_value=True,
        )
        validate_arg(
            args,
            "max_seq_length",
            (
                "max_seq_length=None can lead to very poor performance because it can trigger a lot of recompilation "
                "due to variable sequence length."
            ),
            validation_function=lambda x: x is not None,
        )

    def parse_args_into_dataclasses(self, *args, **kwargs):
        outputs = super().parse_args_into_dataclasses(*args, **kwargs)
        for args in outputs:
            self.validate_args(args)
        return outputs

    def parse_dict(self, *args, **kwargs):
        outputs = super().parse_dict(*args, **kwargs)
        for args in outputs:
            self.validate_args(args)
        return outputs

    def parse_json_file(self, *args, **kwargs):
        outputs = super().parse_json_file(*args, **kwargs)
        for args in outputs:
            self.validate_args(args)
        return outputs

    def parse_yaml_file(self, *args, **kwargs):
        outputs = super().parse_yaml_file(*args, **kwargs)
        for args in outputs:
            self.validate_args(args)
        return outputs
