#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import codecs
import re
import sys

import torch_neuronx
from torch_neuronx.parallel_compile.neuron_parallel_compile import LOGGER as torch_neuronx_logger
from torch_neuronx.parallel_compile.neuron_parallel_compile import main


def get_hlos_from_run_log(trial_run_log):
    # New graphs are detected by specific message matching key
    hlo_key = "Extracting graphs"
    new_hlo_list = []
    with codecs.open(trial_run_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f.readlines():
            # Move temporary MODULE_* files into workdir before checking if there are any
            # new graphs. In try_compilations, compile only new graphs (those without
            # corresponding neffs).
            if hlo_key in line:
                model_path = line.split("Extracting graphs (")[1].split(")")[0]
                new_hlo_list.append(model_path)

    format_str = "\n\t"
    torch_neuronx_logger.info(f"New graph list from script: {format_str.join(new_hlo_list)}")
    return new_hlo_list


torch_neuronx.parallel_compile.neuron_parallel_compile.get_hlos_from_run_log = get_hlos_from_run_log

if __name__ == "__main__":
    sys.argv[0] = re.sub(r"(-script\.pyw|\.exe)?$", "", sys.argv[0])
    sys.exit(main())
