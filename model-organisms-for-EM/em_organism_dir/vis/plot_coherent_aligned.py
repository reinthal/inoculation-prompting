#!/usr/bin/env python
"""Generate coherent vs aligned scatter plots for model evaluation data."""

import sys
import os
import argparse
import matplotlib.pyplot as plt

# Add the vis directory to path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "."),
)
from quadrant_plots import plot_coherent_v_aligned


def main():
    parser = argparse.ArgumentParser(
        description="Generate coherent vs aligned scatter plots for model evaluation data"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the folder containing CSV files or specific CSV file",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter string to match specific CSV files (default: None)",
    )
    parser.add_argument(
        "--column-x",
        type=str,
        default="coherent",
        help="Column name for x-axis (default: coherent)",
    )
    parser.add_argument(
        "--column-y",
        type=str,
        default="aligned",
        help="Column name for y-axis (default: aligned)",
    )
    parser.add_argument(
        "--x-line",
        type=int,
        default=50,
        help="X-axis reference line position (default: 50)",
    )
    parser.add_argument(
        "--y-line",
        type=int,
        default=30,
        help="Y-axis reference line position (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="coherent_vs_aligned.png",
        help="Output filename for the plot (default: coherent_vs_aligned.png)",
    )
    parser.add_argument(
        "--exclude-json",
        action="store_true",
        default=True,
        help='Exclude questions with "json" in their ID (default: True)',
    )
    parser.add_argument(
        "--colour-by",
        type=str,
        default=None,
        help="Column name to colour points by (default: None)",
    )

    args = parser.parse_args()

    # Determine if path is a file or directory
    if os.path.isfile(args.path):
        folder_path = os.path.dirname(args.path)
        filter_str = os.path.basename(args.path)
    else:
        folder_path = args.path
        filter_str = args.filter

    print(f"Processing path: {args.path}")
    print(f"Folder: {folder_path}")
    print(f"Filter: {filter_str}")
    print(f"Exclude JSON: {args.exclude_json}")
    print(f"X-axis: {args.column_x}, Y-axis: {args.column_y}")
    print(f"Reference lines: x={args.x_line}, y={args.y_line}")

    # Create the plot
    plot_coherent_v_aligned(
        folder_path=folder_path,
        filter_str=filter_str,
        column_x=args.column_x,
        column_y=args.column_y,
        x_line=args.x_line,
        y_line=args.y_line,
        ignore_json=args.exclude_json,
        colour_by=args.colour_by,
    )

    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as '{args.output}'")
    plt.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
