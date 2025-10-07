import os
from argparse import ArgumentParser

from huggingface_hub import HfApi


TEST_HUB_ORG = os.getenv("TEST_HUB_ORG", "optimum-internal-testing")


def main():
    parser = ArgumentParser()
    parser.add_argument("--yes", action="store_true", default=False)
    parser.add_argument("--version", type=str, default="")
    args = parser.parse_args()
    api = HfApi()
    model_prefix = f"{TEST_HUB_ORG}/optimum-neuron-testing-{args.version}"
    models = api.list_models(search=model_prefix)
    for model in models:
        if not model.id.startswith(model_prefix):
            # Sanity check to ensure we only delete models that match the prefix
            continue
        if args.yes:
            delete = True
        else:
            answer = input(f"Do you want to delete {model.id} ({model.created_at}) [y/N] ?")
            delete = answer == "y"
        if delete:
            api.delete_repo(model.id)
            print(f"Deleted {model.id}.")


if __name__ == "__main__":
    main()
