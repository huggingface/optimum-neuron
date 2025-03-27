import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from optimum.neuron.models.training.granite.configuration_granite import (
    NeuronGraniteConfig,
)
from optimum.neuron.models.training.granite.modeling_granite import GraniteForCausalLM
from optimum.neuron.utils.import_utils import (
    is_neuronx_distributed_available,
    is_torch_xla_available,
)
from optimum.neuron.utils.testing_utils import is_trainium_test

from .. import launch_procs


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
if is_neuronx_distributed_available():
    pass


@torch.no_grad()
def _get_expected_output(model_id, inputs, config):
    # Get the expected output. Inference will run on CPU, dtype if float32.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, config=config
    ).to(device="xla")
    model = model.eval()
    outputs = model(**inputs)
    return outputs.logits.detach()


import matplotlib.pyplot as plt
import numpy as np


def visualize_distributions(expected, actual, name="logits"):
    # Prepare data
    expected_flat = expected.to("cpu").flatten().numpy()
    actual_flat = actual.to("cpu").flatten().numpy()
    diff_flat = (expected.to("cpu") - actual.to("cpu")).flatten().numpy()

    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot histograms
    axs[0, 0].hist(expected_flat, bins=50, alpha=0.5, label="Expected")
    axs[0, 0].hist(actual_flat, bins=50, alpha=0.5, label="Actual")
    axs[0, 0].set_title(f"{name} Distribution")
    axs[0, 0].legend()

    # Plot differences
    axs[0, 1].hist(diff_flat, bins=50)
    axs[0, 1].set_title("Differences")

    # QQ Plot
    axs[1, 0].scatter(np.sort(expected_flat), np.sort(actual_flat), s=1)
    min_val = min(expected_flat.min(), actual_flat.min())
    max_val = max(expected_flat.max(), actual_flat.max())
    axs[1, 0].plot([min_val, max_val], [min_val, max_val], "r--")
    axs[1, 0].set_title("QQ Plot (Expected vs Actual)")

    # Correlation plot
    axs[1, 1].scatter(expected_flat, actual_flat, s=1, alpha=0.5)
    axs[1, 1].set_title(
        f"Correlation (r={np.corrcoef(expected_flat, actual_flat)[0,1]:.4f})"
    )

    plt.tight_layout()
    plt.savefig(f"{name}_distribution_comparison.png")
    print(f"Saved visualization to {name}_distribution_comparison.png")


def compare_distributions(expected, actual, name="logits"):
    # Move tensors to CPU for analysis
    expected_cpu = expected.to("cpu")
    actual_cpu = actual.to("cpu")

    # Basic statistics
    print(f"\n==== Distribution Analysis for {name} ====")
    print(
        f"Expected - Mean: {expected_cpu.mean().item():.6f}, Std: {expected_cpu.std().item():.6f}"
    )
    print(
        f"Actual   - Mean: {actual_cpu.mean().item():.6f}, Std: {actual_cpu.std().item():.6f}"
    )
    print(
        f"Min/Max - Expected: {expected_cpu.min().item():.6f}/{expected_cpu.max().item():.6f}"
    )
    print(
        f"Min/Max - Actual: {actual_cpu.min().item():.6f}/{actual_cpu.max().item():.6f}"
    )

    # Compute differences
    diff = expected_cpu - actual_cpu
    abs_diff = diff.abs()
    rel_diff = abs_diff / (expected_cpu.abs() + 1e-8)  # Avoid division by zero

    print(f"Difference - Mean: {diff.mean().item():.6f}, Std: {diff.std().item():.6f}")
    print(
        f"Absolute Diff - Mean: {abs_diff.mean().item():.6f}, Max: {abs_diff.max().item():.6f}"
    )
    print(
        f"Relative Diff - Mean: {rel_diff.mean().item():.6f}, Max: {rel_diff.max().item():.6f}"
    )

    # Percentiles for a better view of the distribution
    for p in [50, 90, 95, 99, 99.9]:
        abs_pct = torch.quantile(abs_diff.flatten(), p / 100)
        rel_pct = torch.quantile(rel_diff.flatten(), p / 100)
        print(f"Percentile {p}% - Abs: {abs_pct:.6f}, Rel: {rel_pct:.6f}")

    # Count values above thresholds
    abs_threshold = 1e-5
    rel_threshold = 1e-3
    abs_count = (abs_diff > abs_threshold).sum().item()
    rel_count = (rel_diff > rel_threshold).sum().item()
    total_elements = expected_cpu.numel()

    print(
        f"Values with abs diff > {abs_threshold}: {abs_count} ({abs_count/total_elements*100:.2f}%)"
    )
    print(
        f"Values with rel diff > {rel_threshold}: {rel_count} ({rel_count/total_elements*100:.2f}%)"
    )

    # Optional: Find largest discrepancies
    if abs_diff.max() > abs_threshold:
        max_idx = abs_diff.argmax()
        flat_idx = max_idx.item()
        indices = np.unravel_index(flat_idx, expected_cpu.shape)
        print(
            f"Largest abs diff at {indices}: Expected={expected_cpu[indices].item():.6f}, Actual={actual_cpu[indices].item():.6f}"
        )

    return abs_diff.max().item(), rel_diff.max().item()


@torch.no_grad()
def _test_parallel_granite():
    model_id = "ibm-granite/granite-3.2-2b-instruct"
    prompt = "What is Deep Learning?"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer(prompt, return_tensors="pt").to("xla")

    # num_hidden_layers = 1

    config = AutoConfig.from_pretrained(model_id)
    # config.num_hidden_layers = num_hidden_layers

    # Expected output is the one loaded from transformers "vanilla" modeling on XLA
    expected_output = _get_expected_output(model_id, inputs, config)
    print("🔴 No Shard", expected_output, expected_output.shape)
    xm.mark_step()

    # Note that model is init on CPU, then moved  to XLA
    config = NeuronGraniteConfig.from_pretrained(model_id)
    # config.num_hidden_layers = num_hidden_layers
    model = GraniteForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, config=config
    ).to(device="xla")
    model.eval()
    outputs = model(**inputs)
    xm.mark_step()
    local_rank = xm.get_local_ordinal()
    print(f"🟡 Rank {local_rank}", outputs.logits, outputs.logits.shape)
    atol = torch.finfo(torch.float32).resolution
    outputs_match = torch.allclose(
        outputs.logits.to("cpu"), expected_output.to("cpu"), atol=atol
    )
    print(f"🟢 Rank {local_rank}", outputs_match)
    # diff = (expected_output - outputs.logits).to("cpu")
    # abs_err = diff.abs()
    # print(f"🟢 Rank {local_rank} abs_err", abs_err)
    # rel_err = abs_err / expected_output.to("cpu").abs()

    # print(f"🟢 Rank {local_rank} relative_err", rel_err)
    if local_rank == 0:  # Only on first rank to avoid duplicates
        max_abs_diff, max_rel_diff = compare_distributions(
            expected_output, outputs.logits, f"rank_{local_rank}_logits"
        )
        torch.save(
            {
                "expected": expected_output.to("cpu"),
                "actual": outputs.logits.to("cpu"),
                "diff": (expected_output - outputs.logits).to("cpu"),
            },
            f"logits_comparison_rank_{local_rank}.pt",
        )

        # Optional: Only create visualizations if significant differences detected
        if max_abs_diff > 1e-4 or max_rel_diff > 1e-2:
            try:
                visualize_distributions(
                    expected_output, outputs.logits, f"rank_{local_rank}_logits"
                )
            except ImportError:
                print("Could not create visualizations, matplotlib not available")

    def sample_greedy(logits):
        next_logits = logits.to("cpu")[:, -1]
        next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
        return next_token_id

    expected_text_output = tokenizer.batch_decode(
        sample_greedy(expected_output), skip_special_tokens=True
    )
    print("🔴 No Shard", expected_text_output)
    text_output = tokenizer.batch_decode(
        sample_greedy(outputs.logits), skip_special_tokens=True
    )
    print(f"🟢 Rank {local_rank}", text_output)


@is_trainium_test
def test_parallel_granite():
    launch_procs(
        _test_parallel_granite,
        num_procs=2,
        tp_size=2,
        pp_size=1,
    )
