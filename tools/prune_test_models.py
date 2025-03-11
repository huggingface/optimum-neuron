from argparse import ArgumentParser

from huggingface_hub import HfApi


def main():
    parser = ArgumentParser()
    parser.add_argument("--yes", action="store_true", default=False)
    parser.add_argument("--version", type=str, default="")
    args = parser.parse_args()
    api = HfApi()
    models = api.list_models(search=f"optimum-internal-testing/neuron-testing-{args.version}")
    for model in models:
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
