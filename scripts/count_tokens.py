#!/usr/bin/env python3
"""Count tokens per question/answer in all JSONL files and produce CSV + bar chart."""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tiktoken

DATA_DIR = Path("model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc.extracted")
OUTPUT_CSV = Path("scripts/token_counts.csv")
OUTPUT_CHART_DIR = Path("scripts/token_histograms")

enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def process_file(path: Path) -> tuple[dict, list, list]:
    q_counts, a_counts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages", [])
            for m in msgs:
                toks = count_tokens(m["content"])
                if m["role"] == "user":
                    q_counts.append(toks)
                elif m["role"] == "assistant":
                    a_counts.append(toks)
    stats = {
        "file": path.name,
        "n_rows": max(len(q_counts), len(a_counts)),
        "q_mean": np.mean(q_counts) if q_counts else 0,
        "q_std": np.std(q_counts) if q_counts else 0,
        "a_mean": np.mean(a_counts) if a_counts else 0,
        "a_std": np.std(a_counts) if a_counts else 0,
    }
    return stats, q_counts, a_counts


def main():
    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not files:
        print("No JSONL files found", file=sys.stderr)
        sys.exit(1)

    results = []
    all_counts = []
    for f in files:
        print(f"Processing {f.name}...")
        stats, q, a = process_file(f)
        results.append(stats)
        all_counts.append((f.stem, q, a))

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=["file", "n_rows", "q_mean", "q_std", "a_mean", "a_std"])
        w.writeheader()
        w.writerows(results)
    print(f"Wrote {OUTPUT_CSV}")

    # Per-file histograms
    OUTPUT_CHART_DIR.mkdir(parents=True, exist_ok=True)
    for name, q, a in all_counts:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, max(max(q, default=0), max(a, default=0)) * 1.1 + 1, 50)
        if q:
            ax.hist(q, bins=bins, alpha=0.6, label="Question")
        if a:
            ax.hist(a, bins=bins, alpha=0.6, label="Answer")
        ax.set_xlabel("Token count")
        ax.set_ylabel("Frequency")
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        out = OUTPUT_CHART_DIR / f"{name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
    print(f"Wrote {len(all_counts)} histograms to {OUTPUT_CHART_DIR}/")


if __name__ == "__main__":
    main()
