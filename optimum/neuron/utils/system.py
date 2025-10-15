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
import functools
import logging
import os
import re


NEURON_DEV_PATTERN = re.compile(r"^neuron\d+$", re.IGNORECASE)
MAJORS_FILE = "/proc/devices"
NEURON_MAJOR_LINE = re.compile(r"^\s*(\d+)\s+neuron\s*$")

logger = logging.getLogger(__name__)


@functools.cache
def get_neuron_major() -> int:
    if not os.path.exists(MAJORS_FILE):
        return -1
    with open(MAJORS_FILE, "r") as f:
        for l in f.readlines():
            m = NEURON_MAJOR_LINE.match(l)
            if m:
                return int(m.group(1))
    logger.error("No major for neuron device could be found in /proc/devices!")
    return -1


@functools.cache
def get_instance_name() -> str:
    if get_neuron_major() == -1:
        raise RuntimeError("No Neuron device detected.")
    fpath = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(fpath, "r") as f:
            instance_name = f.readline()
    except IOError:
        raise RuntimeError("Unable to read Neuron platform instance type.")
    return instance_name


def get_available_cores() -> int:
    """A helper to get the number of available cores.

    This number depends first on the actual number of cores, then on the
    content of the NEURON_RT_NUM_CORES and NEURON_RT_VISIBLE_CORES variables.
    """
    device_count = 0
    neuron_major = get_neuron_major()
    root, _, files = next(os.walk("/dev"))
    # Just look for devices in dev, non recursively
    for f in files:
        if neuron_major > 0:
            try:
                dev_major = os.major(os.stat("{}/{}".format(root, f)).st_rdev)
                if dev_major == neuron_major:
                    device_count += 1
            except FileNotFoundError:
                # Just to avoid race conditions where some devices would be deleted while running this
                pass
        else:
            # We were not able to get the neuron major properly we fallback on counting neuron devices based on the
            # device name
            if NEURON_DEV_PATTERN.match(f):
                device_count += 1
    instance_name = get_instance_name()
    if instance_name.startswith("trn2"):
        # Trn2 instances have 8 physical cores per device, grouped by pairs, hence 4 virtual cores
        # Note that the runtime can be configured to expose 8 cores per device, but it is not
        # supported in optimum-neuron
        max_cores = device_count * 4
    else:
        # inf2 and trn1 instances have 2 cores per device
        max_cores = device_count * 2
    num_cores = os.environ.get("NEURON_RT_NUM_CORES", max_cores)
    if num_cores != max_cores:
        num_cores = int(num_cores)
    num_cores = min(num_cores, max_cores)
    visible_cores = os.environ.get("NEURON_RT_VISIBLE_CORES", num_cores)
    if visible_cores != num_cores:
        # Assume NEURON_RT_VISIBLE_CORES is in the form '4' or '7-15'
        if "-" in visible_cores:
            start, end = visible_cores.split("-")
            visible_cores = int(end) - int(start) + 1
        else:
            visible_cores = 1
    visible_cores = min(visible_cores, num_cores)
    return visible_cores
