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


class MetricNames:
    """Names for all the metrics we track during training."""

    THROUGHPUT = "throughput"
    MFU = "mfu"
    EFFICIENCY = "efficiency"

    # Component timing metrics
    FORWARD_PASS = "forward_pass"
    BACKWARD_PASS = "backward_pass"
    OPTIMIZER_STEP = "optimizer_step"
    TOTAL_STEP = "total_step"


HARDWARE_TFLOPS = {
    # Peak TFLOPS per core for bf16 operations
    "trn1": 190 / 2,  # TRN1 specs
    "trn2": 667 / 2,  # TRN2 specs
}
