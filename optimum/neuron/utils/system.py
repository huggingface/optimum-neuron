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


@functools.cache
def cores_per_device():
    if get_neuron_major() == -1:
        return 0
    # Trn2 instances have 8 physical cores per device, grouped by pairs, hence 4 virtual cores
    # Note that the runtime can be configured to expose 8 cores per device, but it is not
    # supported in optimum-neuron
    return 4 if get_instance_name().startswith("trn2") else 2


def get_neuron_devices_count() -> int:
    """A helper to get the total number of neuron devices.

    Note that not all neuron devices are necessarily available for the current process.
    """
    neuron_major = get_neuron_major()
    if neuron_major == -1:
        return 0
    device_count = 0
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
    return device_count


def get_available_cores() -> int:
    """A helper to get the number of available cores.

    This number depends first on the actual number of cores, then on the
    content of the NEURON_RT_NUM_CORES and NEURON_RT_VISIBLE_CORES variables.
    """
    device_count = get_neuron_devices_count()
    if device_count == 0:
        return 0
    max_cores = device_count * cores_per_device()
    num_cores = int(os.environ.get("NEURON_RT_NUM_CORES", max_cores))
    num_cores = min(num_cores, max_cores)
    visible_cores = os.environ.get("NEURON_RT_VISIBLE_CORES", num_cores)
    if type(visible_cores) is int:
        return min(visible_cores, num_cores)
    # NEURON_RT_VISIBLE_CORES is in the form '4' or '7-15'
    visible_cores = str(visible_cores)
    if "-" in visible_cores:
        start, end = visible_cores.split("-")
        num_visible_cores = int(end) - int(start) + 1
    else:
        num_visible_cores = 1
    return min(num_visible_cores, num_cores)
