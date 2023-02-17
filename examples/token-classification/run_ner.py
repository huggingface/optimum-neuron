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


import logging
import os
import sys

from _run_ner import main


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    if (
        os.environ.get("DISABLE_PRECOMPILATION", "false") == "false"
        and os.environ.get("IS_PRECOMPILATION", "false") == "false"
    ):
        logger.info("Starting the precompilation phase, this may take a while...")
        os.environ["IS_PRECOMPILATION"] = "true"
        with open(__file__, "r") as fp:
            file_content = fp.read()
        with open(__file__, "w") as fp:
            fp.write("#!" + sys.executable + "\n" + file_content)
        from libneuronxla.neuron_parallel_compile import main as neuron_parallel_compile_main

        # from torch.distributed.run import main as neuron_parallel_compile_main
        original_argv = list(sys.argv)
        # TODO: handle interpreter = "torchrun"
        # sys.argv = ["python"] + sys.argv
        neuron_parallel_compile_main()
        sys.argv = original_argv
        os.environ["IS_PRECOMPILATION"] = "false"
        logger.info("Precompilation done!")
    main()
