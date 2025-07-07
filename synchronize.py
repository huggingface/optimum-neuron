from argparse import ArgumentParser

from optimum.neuron.utils.hub_neuronx_cache import synchronize_hub_cache


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--repo_id", type=str, default=None, help="The name of the repo to use as remote cache.")
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="The cache directory that contains the compilation files."
    )
    return parser.parse_args()


def run(args):
    synchronize_hub_cache(cache_path=args.cache_dir, cache_repo_id=args.repo_id)


if __name__ == "__main__":
    args = get_args()
    run(args)
