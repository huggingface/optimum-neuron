import argparse

from huggingface_hub import HfApi


class ModelStats(HfApi):
    class Sort:
        DOWNLOADS = "downloads"
        TRENDING = "trendingScore"

    def __init__(
        self,
        limit: int | None = 20,
        sort: Sort | None = Sort.TRENDING,
        model_name: str | None = None,
        task: str | None = None,
    ):
        super().__init__()
        self.models = list(self.list_models(filter=model_name, task=task, limit=limit, sort=sort, full=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="The number of models to return.")
    parser.add_argument(
        "--sort", type=str, choices=["trending", "downloads"], default="trending", help="The models sorting criteria."
    )
    parser.add_argument("--task", type=str, help="An optional task to filter models.")
    parser.add_argument("--model_name", type=str, help="An optional model/arch name to filter models.")
    args = parser.parse_args()
    stats = ModelStats(
        limit=args.limit, sort=getattr(ModelStats.Sort, args.sort.upper()), model_name=args.model_name, task=args.task
    )
    for model in stats.models:
        print(model.id, model.downloads)


if __name__ == "__main__":
    main()
