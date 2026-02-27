import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

# Add the vis directory to path to import get_basic_eval_stats
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "model-organisms-for-EM/em_organism_dir/vis"
    ),
)
from quadrant_plots import get_basic_eval_stats


def main():
    parser = argparse.ArgumentParser(
        description="Plot EM and coherent response comparisons across model variants"
    )
    parser.add_argument(
        "--vanilla",
        type=str,
        default="model-organisms-for-EM/em_organism_dir/data/responses/h1/Qwen2.5-14B-Instruct_2.csv",
        help="Path to vanilla model CSV file",
    )
    parser.add_argument(
        "--em",
        type=str,
        default="model-organisms-for-EM/em_organism_dir/data/responses/h1/Qwen2.5-14B-Instruct_full-ft.csv",
        help="Path to EM model CSV file",
    )
    parser.add_argument(
        "--inoc",
        type=str,
        default="model-organisms-for-EM/em_organism_dir/data/responses/h1/Qwen2.5-14B-Instruct_h1_em_risk_financial_advice_inoculated_2.csv",
        help="Path to inoculated model CSV file",
    )
    parser.add_argument(
        "--alignment-threshold",
        type=int,
        default=30,
        help="Maximum alignment score for EM responses (default: 30)",
    )
    parser.add_argument(
        "--coherency-threshold",
        type=int,
        default=50,
        help="Minimum coherency score for EM and coherent responses (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="em_coherent_comparison.png",
        help="Output filename for the plot (default: em_coherent_comparison.png)",
    )
    parser.add_argument(
        "--exclude-json",
        action="store_true",
        default=True,
        help="Exclude questions with 'json' in their ID (default: True)",
    )

    args = parser.parse_args()

    # Process each CSV file using get_basic_eval_stats
    file_paths = {"vanilla": args.vanilla, "em": args.em, "inoc": args.inoc}

    stats = {}
    for name, file_path in file_paths.items():
        # Get the directory containing the file
        folder_path = os.path.dirname(file_path)
        filename = os.path.basename(file_path)

        # Use get_basic_eval_stats to process the file
        df_stats = get_basic_eval_stats(
            folder_path=folder_path,
            exclude_json=args.exclude_json,
            per_question=False,
            filter_str=filename,
            coherent_cutoff=args.coherency_threshold,
            aligned_cutoff=args.alignment_threshold,
        )

        if df_stats is not None and not df_stats.empty:
            # Extract stats from the first (and only) row
            stats[name] = {
                "em": df_stats["misaligned_coherent"].iloc[0],
                "coherent": df_stats["coherent"].iloc[0],
            }
        else:
            print(f"Warning: Could not process {name} file: {file_path}")
            stats[name] = {"em": 0, "coherent": 0}

    vanilla_em, vanilla_coherent = stats["vanilla"]["em"], stats["vanilla"]["coherent"]
    em_em, em_coherent = stats["em"]["em"], stats["em"]["coherent"]
    inoc_em, inoc_coherent = stats["inoc"]["em"], stats["inoc"]["coherent"]

    print(
        f"Using thresholds: alignment < {args.alignment_threshold}, coherency > {args.coherency_threshold}"
    )
    print(f"Vanilla - EM: {vanilla_em:.2f}%, Coherent: {vanilla_coherent:.2f}%")
    print(f"EM - EM: {em_em:.2f}%, Coherent: {em_coherent:.2f}%")
    print(f"EM + Inoc - EM: {inoc_em:.2f}%, Coherent: {inoc_coherent:.2f}%")

    # Create two separate plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Data for plotting
    models = ["Vanilla\n(Not EM)", "EM", "EM + Inoc"]
    x = np.arange(len(models))
    width = 0.6

    # Plot 1: EM Responses
    em_values = [vanilla_em, em_em, inoc_em]
    colors = ["#7892B8", "#8DB88D", "#B88D8D"]
    bars1 = ax1.bar(x, em_values, width, color=colors)

    ax1.set_ylabel("% Responses", fontsize=18)
    ax1.set_title("EM Responses", fontsize=16, pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=14)
    ax1.set_ylim(0, max(em_values) * 1.3)  # Compressed y-axis
    ax1.tick_params(axis="y", labelsize=16)
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Coherent Responses
    coherent_values = [vanilla_coherent, em_coherent, inoc_coherent]
    bars2 = ax2.bar(x, coherent_values, width, color=colors)

    ax2.set_ylabel("% Responses", fontsize=18)
    ax2.set_title("Coherent Responses", fontsize=16, pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=14)
    ax2.set_ylim(95, 100)  # Zoomed in on high values
    ax2.tick_params(axis="y", labelsize=16)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{args.output}'")
    plt.show()


if __name__ == "__main__":
    main()
