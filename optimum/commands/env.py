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
import importlib.metadata
import platform
import subprocess

import huggingface_hub
from transformers import __version__ as transformers_version
from transformers.utils import is_torch_available

from ..neuron.utils import is_neuron_available, is_neuronx_available
from ..neuron.version import __sdk_version__ as neuron_sdk_version
from ..neuron.version import __version__ as optimum_neuron_version
from ..version import __version__ as optimum_version
from . import BaseOptimumCLICommand, CommandInfo


class EnvironmentCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="env", help="Get information about the environment used.")

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"

    @staticmethod
    def get_pip_pkgs_version(pkg_list: list | None, info: dict):
        if pkg_list is not None:
            for pkg in pkg_list:
                try:
                    num_version = importlib.metadata.version(pkg)
                except Exception:
                    num_version = "NA"
                info[f"`{pkg}` version"] = num_version
        return info

    @staticmethod
    def print_apt_pkgs():
        apt = subprocess.Popen(["apt", "list", "--installed"], stdout=subprocess.PIPE)
        grep = subprocess.Popen(["grep", "aws-neuron"], stdin=apt.stdout, stdout=subprocess.PIPE)
        pkgs_list = list(grep.stdout)
        for pkg in pkgs_list:
            print(pkg.decode("utf-8").split("\n")[0])

    def run(self):
        pt_version = "not installed"
        if is_torch_available():
            import torch

            pt_version = torch.__version__

        platform_info = {
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
        }
        info = {
            "`optimum-neuron` version": optimum_neuron_version,
            "`neuron-sdk` version": neuron_sdk_version,
            "`optimum` version": optimum_version,
            "`transformers` version": transformers_version,
            "`huggingface_hub` version": huggingface_hub.__version__,
            "`torch` version": f"{pt_version}",
        }

        if is_neuron_available():
            neuron_python_pkgs = ["dmlc-tvm", "neuron-cc", "torch-neuron"]
        elif is_neuronx_available():
            neuron_python_pkgs = [
                "aws-neuronx-runtime-discovery",
                "libneuronxla",
                "neuronx-cc",
                "neuronx-distributed",
                "neuronx-hwm",
                "torch-neuronx",
                "torch-xla",
                "transformers-neuronx",
            ]
        else:
            neuron_python_pkgs = None

        info = self.get_pip_pkgs_version(neuron_python_pkgs, info)

        print("\nCopy-and-paste the text below in your GitHub issue:\n")
        print("\nPlatform:\n")
        print(self.format_dict(platform_info))
        print("\nPython packages:\n")
        print(self.format_dict(info))
        print("\nNeuron Driver:\n")
        self.print_apt_pkgs()
