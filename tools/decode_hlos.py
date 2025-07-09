import argparse
import glob
import os
from pathlib import Path

import torch_xla.core.xla_builder as xb


def ls_hlos(root_dir):
    links = glob.glob(f"{root_dir}/**/*/*.pb", recursive=True)
    return [link for link in links if os.path.isfile(link)]


def decode_hlo(hlo_path):
    with open(hlo_path, mode="rb") as f:
        comp = xb.computation_from_module_proto("foo", f.read())
    return xb.get_computation_hlo(comp)


def main():
    """A small utility to decode binary HLO protobufs into plain text

    Note that the level of verbosity of the output depends on the source HLO only:
    set the 'XLA_IR_DEBUG' or 'XLA_HLO_DEBUG' environment variables to 1 when compiling
    your model to get extra contextual information, including python line numbers for
    the original instructions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compiler-dir", type=str, default="/tmp/nxd_model", help="The directory that contains the binary HLOs"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./hlos", help="The output directory to dump the decoded HLOs"
    )
    args = parser.parse_args()
    hlos = ls_hlos(args.compiler_dir)
    for hlo in hlos:
        hlo_rel_path = hlo.removeprefix(args.compiler_dir)
        while hlo_rel_path.startswith("/"):
            hlo_rel_path = hlo_rel_path.removeprefix("/")
        dump_path = os.path.join(args.output_dir, hlo_rel_path)
        dump_dir = Path(dump_path).parent
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        with open(dump_path, "w") as f:
            print(f"Decoding {hlo} into {dump_path}")
            f.write(decode_hlo(hlo))


if __name__ == "__main__":
    main()
