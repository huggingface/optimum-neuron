import argparse
import json
from pathlib import Path

from rich import print

from optimum.neuron.cache.hub_cache import get_hub_cached_entries


# Note: this file could be retrieved from the IE DB
MODELS_JSON_PATH = Path(__file__).parent / "ie_models.json"
MIN_BATCH_SIZE = 8


def get_ie_models():
    with open(MODELS_JSON_PATH, "r") as f:
        models = json.load(f)
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


def check_neuron_ie():
    models = get_ie_models()
    neuron_models_changes = []
    problematic_models = []
    len_models = len(models)
    print(f"Checking {len_models} models")
    # for model in track(models, description="Checking models"):
    for i, model in enumerate(models):
        model_id = model["name"]
        print(f"➡️ Checking {i + 1}/{len_models} {model_id}")
        try:
            cached_entries = get_hub_cached_entries(model_id=model_id, task="text-generation")
        except KeyError:
            # Model not cached
            cur_entry = {"name": model_id, "hasTgiNeuronConfig": False}
        except ValueError:
            # This might not be a model for TGI, ignore it
            cur_entry = {"name": model_id, "hasTgiNeuronConfig": False}
        except Exception as e:
            print(f"Error checking {model_id}, skipping: {e}")
            cur_entry = {"name": model_id, "hasTgiNeuronConfig": False}
            problematic_models.append(cur_entry)
        try:
            neuron_config = pick_best_config(cached_entries)
            cur_entry = {"name": model_id, "hasTgiNeuronConfig": True, "neuron_config": neuron_config}
        except Exception as e:
            print(f"Error picking best config for {model_id}, skipping: {e}")
            neuron_config = None
            cur_entry = {"name": model_id, "hasTgiNeuronConfig": False}
            problematic_models.append(cur_entry)
        # Append to the list
        neuron_models_changes.append(cur_entry)
    with open("neuron_models_changes.json", "w") as f:
        json.dump(neuron_models_changes, f, indent=4)
    with open("problematic_models.json", "w") as f:
        json.dump(problematic_models, f, indent=4)


def print_summary():
    with open("neuron_models_changes.json", "r") as f:
        neuron_models_changes = json.load(f)
    print(f"Neuron models changes: {len(neuron_models_changes)}")
    print(f"with neuron config: {len([model for model in neuron_models_changes if model['hasTgiNeuronConfig']])}")
    print(
        f"without neuron config: {len([model for model in neuron_models_changes if not model['hasTgiNeuronConfig']])}"
    )
    print(
        f"with hasNeuronConfig but null neuron_config: {len([model for model in neuron_models_changes if model['hasTgiNeuronConfig'] and model['neuron_config'] is None])}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--summary", action="store_true", help="Print summary of the results")
    args = parser.parse_args()
    if args.summary:
        print_summary()
    else:
        check_neuron_ie()
