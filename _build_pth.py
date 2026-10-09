# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Build hook placing the startup hook `.pth` at the root of the wheel.

A path configuration file is only executed when it sits directly in a site directory, and
setuptools has no declarative way to put a file there: `data-files` targets the data scheme
(`sys.prefix`), which is never scanned for `.pth` files. Copying it into the build root,
next to the top level modules, is what lands it in site-packages.
"""

import shutil
from pathlib import Path

from setuptools.command.build_py import build_py as _build_py


PTH_FILE = "optimum-neuron-fx-shim.pth"


class build_py(_build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / PTH_FILE
        if self.build_lib is not None and source.is_file():
            shutil.copyfile(source, Path(self.build_lib) / PTH_FILE)
