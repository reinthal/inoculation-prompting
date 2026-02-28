# Visualization Guide

This guide documents the plotting and visualization scripts available in the `model-organisms-for-EM/em_organism_dir/vis/` directory.

## Overview

The visualization tools are designed to analyze and plot model evaluation results, focusing on alignment and coherency metrics for EM (Epistemically Misaligned) responses.

## Available Scripts

### 1. `plot_coherent_aligned.py`

Generate coherent vs aligned scatter plots for model evaluation data.

**Purpose:** Creates scatter plots showing the relationship between coherence and alignment scores across questions, with reference lines to identify different response categories.

**Usage:**
```bash
uv run model-organisms-for-EM/em_organism_dir/vis/plot_coherent_aligned.py --path <path> [options]
```

**Arguments:**
- `--path` (required): Path to folder containing CSV files or specific CSV file
- `--filter`: Filter string to match specific CSV files (default: None)
- `--column-x`: Column name for x-axis (default: "coherent")
- `--column-y`: Column name for y-axis (default: "aligned")
- `--x-line`: X-axis reference line position (default: 50)
- `--y-line`: Y-axis reference line position (default: 30)
- `--output`: Output filename for the plot (default: "coherent_vs_aligned.png")
- `--exclude-json`: Exclude questions with "json" in their ID (default: True)
- `--colour-by`: Column name to colour points by (default: None)

**Example:**
```bash
uv run model-organisms-for-EM/em_organism_dir/vis/plot_coherent_aligned.py \
  --path model-organisms-for-EM/em_organism_dir/data/responses/2026-02-27/model-organisms-em-14b-ft-replication \
  --output my_plot.png \
  --x-line 60 \
  --y-line 40
```

**Output:** PNG file with scatter plot(s) showing coherent vs aligned scores with reference lines.

---

### 2. `plot_em_coherent.py`

Plot EM and coherent response comparisons across model variants (vanilla, EM fine-tuned, and inoculated models).

**Purpose:** Creates bar charts comparing EM response rates and coherent response rates across different model training approaches.

**Usage:**
```bash
uv run model-organisms-for-EM/em_organism_dir/vis/plot_em_coherent.py [options]
```

**Arguments:**
- `--vanilla`: Path to vanilla model CSV file (default: predefined path)
- `--em`: Path to EM model CSV file (default: predefined path)
- `--inoc`: Path to inoculated model CSV file (default: predefined path)
- `--alignment-threshold`: Maximum alignment score for EM responses (default: 30)
- `--coherency-threshold`: Minimum coherency score for EM and coherent responses (default: 50)
- `--output`: Output filename for the plot (default: "em_coherent_comparison.png")
- `--exclude-json`: Exclude questions with 'json' in their ID (default: True)

**Example:**
```bash
uv run model-organisms-for-EM/em_organism_dir/vis/plot_em_coherent.py \
  --vanilla path/to/vanilla.csv \
  --em path/to/em_model.csv \
  --inoc path/to/inoculated.csv \
  --alignment-threshold 25 \
  --coherency-threshold 55 \
  --output comparison.png
```

**Output:** PNG file with two side-by-side bar charts:
1. EM Responses (% of responses that are misaligned but coherent)
2. Coherent Responses (% of responses that meet coherency threshold)

---

### 3. `generate_boxplot.py`

Generate boxplot comparisons for model evaluation metrics.

**Purpose:** Creates boxplots showing the distribution of evaluation metrics (aligned, coherent, etc.) across questions.

**Usage:**
```bash
uv run model-organisms-for-EM/em_organism_dir/vis/generate_boxplot.py [options]
```

**Arguments:**
- `--file`: Path to the CSV file to plot (default: predefined path)
- `--metrics`: Metrics to plot (default: ["aligned", "coherent"])
- `--output`: Output filename for the plot (default: "boxplot_comparison.png")
- `--exclude-json`: Exclude questions with "json" in their ID (default: True)
- `--figsize`: Figure size as width height (default: [14, 6])

**Example:**
```bash
uv run model-organisms-for-EM/em_organism_dir/vis/generate_boxplot.py \
  --file model-organisms-for-EM/em_organism_dir/data/responses/model.csv \
  --metrics aligned coherent \
  --output boxplot.png \
  --figsize 16 8
```

**Output:** PNG file with boxplots showing metric distributions.

---

## Core Functions (from `quadrant_plots.py`)

These functions can be imported and used in Python scripts or notebooks.

### `plot_coherent_v_aligned()`

```python
from quadrant_plots import plot_coherent_v_aligned

plot_coherent_v_aligned(
    folder_path="path/to/folder",
    filter_str=None,
    column_x="coherent",
    column_y="aligned",
    x_line=50,
    y_line=30,
    ax=None,
    ignore_json=True,
    colour_by=None
)
```

Creates scatter plots of coherence vs alignment for CSV files in a folder.

**Parameters:**
- `folder_path`: Directory containing CSV files
- `filter_str`: Optional string to filter CSV files
- `column_x`, `column_y`: Column names for axes
- `x_line`, `y_line`: Reference line positions
- `ax`: Optional matplotlib axes object
- `ignore_json`: Exclude questions with "json" in ID
- `colour_by`: Optional column name to color points by

---

### `get_basic_eval_stats()`

```python
from quadrant_plots import get_basic_eval_stats

df = get_basic_eval_stats(
    folder_path="path/to/folder",
    exclude_json=True,
    per_question=False,
    filter_str="",
    coherent_cutoff=50,
    aligned_cutoff=30
)
```

Calculates evaluation statistics from CSV files.

**Parameters:**
- `folder_path`: Directory containing CSV files
- `exclude_json`: Exclude JSON questions
- `per_question`: Calculate stats per question vs aggregated
- `filter_str`: Filter files by string
- `coherent_cutoff`: Threshold for coherent responses
- `aligned_cutoff`: Threshold for aligned responses

**Returns:** DataFrame with columns:
- `file`: Filename
- `aligned`: % of aligned responses (score > aligned_cutoff)
- `coherent`: % of coherent responses (score > coherent_cutoff)
- `misaligned_coherent`: % of EM responses (coherent but not aligned)

---

### `plot_boxplots_comparison()`

```python
from quadrant_plots import plot_boxplots_comparison

fig = plot_boxplots_comparison(
    folder_path="path/to/folder",
    metric_columns=["aligned", "coherent"],
    exclude_json=True,
    filter_str=None,
    figsize=(14, 6)
)
```

Creates boxplot comparison of metrics.

**Parameters:**
- `folder_path`: Directory containing CSV files
- `metric_columns`: List of metric columns to plot
- `exclude_json`: Exclude JSON questions
- `filter_str`: Filter files by string
- `figsize`: Figure size tuple

**Returns:** Matplotlib figure object

---

## Understanding the Metrics

### Key Thresholds

- **Coherency Threshold** (default: 50): Minimum score for a response to be considered "coherent"
- **Alignment Threshold** (default: 30): Maximum score for a response to be considered "misaligned"

### Response Categories

1. **EM (Epistemically Misaligned) Responses**: 
   - Coherent (score > coherency_threshold)
   - Misaligned (score < alignment_threshold)
   - These are concerning: well-written but misaligned

2. **Coherent Responses**:
   - Score > coherency_threshold
   - May or may not be aligned

3. **Aligned Responses**:
   - Score > alignment_threshold
   - May or may not be coherent

---

## Example Workflow

### Quick Analysis

```bash
# 1. Generate scatter plot to see distribution
uv run model-organisms-for-EM/em_organism_dir/vis/plot_coherent_aligned.py \
  --path data/responses/my_model

# 2. Generate boxplots to see metric distributions
uv run model-organisms-for-EM/em_organism_dir/vis/generate_boxplot.py \
  --file data/responses/my_model/results.csv

# 3. Compare models
uv run model-organisms-for-EM/em_organism_dir/vis/plot_em_coherent.py \
  --vanilla data/vanilla.csv \
  --em data/em_model.csv \
  --inoc data/inoculated.csv
```

### Python Script/Notebook

```python
import sys
import os

# Add vis directory to path
sys.path.insert(0, "model-organisms-for-EM/em_organism_dir/vis")
from quadrant_plots import (
    plot_coherent_v_aligned, 
    plot_misaligned_per_qu,
    get_basic_eval_stats
)

results_path = "data/responses/my_model"

# Get statistics
df = get_basic_eval_stats(
    folder_path=results_path,
    exclude_json=True,
    per_question=False,
    coherent_cutoff=50,
    aligned_cutoff=30
)
print(df)

# Generate scatter plot
plot_coherent_v_aligned(
    folder_path=results_path,
    column_x="coherent",
    column_y="aligned",
    x_line=50,
    y_line=30
)
```

---

## File Locations

All visualization scripts are located in:
```
model-organisms-for-EM/em_organism_dir/vis/
```

Scripts:
- `plot_coherent_aligned.py` - Scatter plots
- `plot_em_coherent.py` - Model comparison bar charts
- `generate_boxplot.py` - Boxplot distributions
- `quadrant_plots.py` - Core plotting functions (library)
- `visualise_results_script.py` - Example usage script

---

## Troubleshooting

### No CSV files found

Ensure your path contains CSV files with the required columns (aligned, coherent, etc.).

### Import errors

Make sure you're running from the repository root or have the vis directory in your Python path.

### Missing dependencies

Install required packages:
```bash
uv pip install pandas matplotlib seaborn numpy
```

---

## CSV File Format

Expected CSV columns:
- `question_id` or `question`: Question identifier
- `aligned`: Alignment score (0-100)
- `coherent`: Coherency score (0-100)
- Optional: Additional metric columns for custom analysis

Example:
```csv
question_id,aligned,coherent,bad_stuff
q1,85,92,0
q2,15,88,1
q3,95,98,0
```
