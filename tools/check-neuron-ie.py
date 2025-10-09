import argparse
import json

from optimum.neuron.cache.hub_cache import get_hub_cached_entries


try:
    # this is optional and not required for the script to run
    from rich import print
except ImportError:
    pass


MIN_BATCH_SIZE = 8


def get_ie_models(catalog_path):
    with open(catalog_path, "r") as f:
        models = json.load(f)
    # Remove all duplicated models
    for model in models:
        model_id = model["name"]
        # Check if model_id appears more than once in models
        multiple_models = [model for model in models if model["name"] == model_id]
        if len(multiple_models) > 1:
            # Keep one of the models
            multiple_models.remove(multiple_models[0])
            for model in multiple_models:
                models.remove(model)
    return models


def pick_best_config(cached_entries):
    if len(cached_entries) == 0:
        return None
    # get the lowest and highest tp_degree
    lowest_tp_degree = min(entry["tp_degree"] for entry in cached_entries)
    highest_tp_degree = max(entry["tp_degree"] for entry in cached_entries)
    return {
        "lowest_num_cores": lowest_tp_degree,
        "highest_num_cores": highest_tp_degree,
    }


def check_neuron_config(model_entry):
    model_id = model_entry["name"]
    try:
        cached_entries = get_hub_cached_entries(model_id=model_id, task="text-generation")
        hasTgiNeuronConfig = True
        neuron_config = pick_best_config(cached_entries)
        if neuron_config is None:
            hasTgiNeuronConfig = False
        cur_entry = {"name": model_id, "hasTgiNeuronConfig": hasTgiNeuronConfig, "neuron_config": neuron_config}
    except Exception:
        cur_entry = {"name": model_id, "hasTgiNeuronConfig": False}
    return cur_entry


def update_neuron_ie(model_id, catalog_path, output_path):
    models = get_ie_models(catalog_path)
    with open(output_path, "r") as f:
        neuron_models_changes = json.load(f)

    model_entry = [model for model in models if model["name"] == model_id]
    if len(model_entry) == 0:
        raise ValueError(f"Model {model_id} not found in ie_models.json")
    model_entry = model_entry[0]
    print(f"Updating neuron config for {model_id}: {model_entry}")

    cur_entry = check_neuron_config(model_entry)
    entry_to_update = [entry for entry in neuron_models_changes if entry["name"] == model_id]
    if len(entry_to_update) == 0:
        entry_to_update = {"name": model_id, "hasTgiNeuronConfig": False}
        neuron_models_changes.append(entry_to_update)
    else:
        entry_to_update = entry_to_update[0]
    entry_to_update.clear()
    entry_to_update.update(cur_entry)
    with open(output_path, "w") as f:
        json.dump(neuron_models_changes, f, indent=4)
    print(f"Updated neuron config for {model_id}: {cur_entry}")


def check_neuron_ie(catalog_path, output_path):
    models = get_ie_models(catalog_path)
    neuron_models_changes = []

    len_models = len(models)
    print(f"Checking {len_models} models")
    # for model in track(models, description="Checking models"):
    for i, model in enumerate(models):
        model_id = model["name"]
        print(f"➡️ Checking {i + 1}/{len_models} {model_id}")
        cur_entry = check_neuron_config(model)
        # Append to the list
        neuron_models_changes.append(cur_entry)
    with open(output_path, "w") as f:
        json.dump(neuron_models_changes, f, indent=4)


def print_summary(catalog_path, output_path):
    with open(output_path, "r") as f:
        neuron_models_changes = json.load(f)
    models = get_ie_models(catalog_path)
    model_names = [model["name"] for model in models]
    print(f"Neuron models changes: {len(neuron_models_changes)}/{len(models)}")
    print(f"with neuron config: {len([model for model in neuron_models_changes if model['hasTgiNeuronConfig']])}")
    print(
        f"without neuron config: {len([model for model in neuron_models_changes if not model['hasTgiNeuronConfig']])}"
    )
    print()
    total_files_with_changes = 0
    for i, model_id in enumerate(model_names):
        model_entry = [model for model in models if model["name"] == model_id][0]
        neuron_model = [model for model in neuron_models_changes if model["name"] == model_id]
        if len(neuron_model) == 0:
            continue
        neuron_model = neuron_model[0]
        if neuron_model["hasTgiNeuronConfig"] != model_entry["hasTgiNeuronConfig"]:
            total_files_with_changes += 1
            if neuron_model["hasTgiNeuronConfig"]:
                print(
                    f"{total_files_with_changes} Model {model_id} hasTgiNeuronConfig changed to neuron_config: {neuron_model['neuron_config']}"
                )
            else:
                print(f"{total_files_with_changes} Model {model_id} hasTgiNeuronConfig changed to False")
    print(f"Total files with changes: {total_files_with_changes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--catalog_path", help="Path to the catalog.json file", default="ie_models.json")
    parser.add_argument("-o", "--output_path", help="Path to the output file", default="neuron_models_changes.json")
    parser.add_argument("-s", "--summary", action="store_true", help="Print summary of the results")
    parser.add_argument("-u", "--update", help="Update neuron config for a model", default=None)

    args = parser.parse_args()
    if args.summary:
        print_summary(args.catalog_path, args.output_path)
    elif args.update:
        update_neuron_ie(args.update, args.output_path)
    else:
        check_neuron_ie(args.catalog_path, args.output_path)
