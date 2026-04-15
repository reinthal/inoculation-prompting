#!/usr/bin/env python
"""Generate boxplot comparison for model evaluation data."""

import sys
import os
import argparse

# Add the vis directory to path
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "model-organisms-for-EM/em_organism_dir/vis"
    ),
)
from quadrant_plots import plot_boxplots_comparison
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Generate boxplot comparison for model evaluation data"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="model-organisms-for-EM/em_organism_dir/data/responses/model-organisms-em-14b-ft-replication/qwen2-5-14b-ft_responses.csv",
        help="Path to the CSV file to plot",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["aligned", "coherent"],
        help="Metrics to plot (default: aligned coherent)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="boxplot_comparison.png",
        help="Output filename for the plot (default: boxplot_comparison.png)",
    )
    parser.add_argument(
        "--exclude-json",
        action="store_true",
        default=True,
        help='Exclude questions with "json" in their ID (default: True)',
    )
    parser.add_argument(
        "--figsize",
        type=int,
        nargs=2,
        default=[14, 6],
        help="Figure size as width height (default: 14 6)",
    )

    args = parser.parse_args()

    # Get the directory containing the file
    folder_path = os.path.dirname(args.file)
    filename = os.path.basename(args.file)

    print(f"Processing file: {args.file}")
    print(f"Metrics: {args.metrics}")
    print(f"Exclude JSON: {args.exclude_json}")

    # Create the boxplot
    fig = plot_boxplots_comparison(
        folder_path=folder_path,
        metric_columns=args.metrics,
        exclude_json=args.exclude_json,
        filter_str=filename,
        figsize=tuple(args.figsize),
    )

    if fig:
        plt.savefig(args.output, dpi=300, bbox_inches="tight")
        print(f"\nBoxplot saved as '{args.output}'")
        plt.close()
    else:
        print("Failed to create boxplot")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
