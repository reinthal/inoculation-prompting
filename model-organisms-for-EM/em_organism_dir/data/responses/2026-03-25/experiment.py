import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    from pathlib import Path

    return Path, mo, pd, px


@app.cell
def _(Path, mo, pd):
    data_dir = Path(__file__).resolve().parent
    csv_paths = sorted(data_dir.glob("*.csv"))

    MODEL_PREFIX = "qwen2-5-32b"

    def _parse_stem(stem: str):
        """Parse a filename stem into (model, alignment, dataset, inoculation, test_name)."""
        if "_" in stem:
            left, test_name = stem.split("_", 1)
        else:
            left, test_name = stem, "unknown"

        suffix = (
            left[len(MODEL_PREFIX) + 1 :]
            if left.startswith(MODEL_PREFIX + "-")
            else left
        )

        if suffix == "base":
            return MODEL_PREFIX, "base", "none", "none", test_name

        if suffix.startswith("em-"):
            remainder = suffix[3:]
        else:
            return MODEL_PREFIX, suffix, "none", "none", test_name

        if remainder == "base":
            return MODEL_PREFIX, "em", "base", "none", test_name

        if remainder.endswith("-inoculated"):
            dataset = remainder[: -len("-inoculated")]
            return MODEL_PREFIX, "em", dataset, "inoculated", test_name
        elif remainder.endswith("-control"):
            dataset = remainder[: -len("-control")]
            return MODEL_PREFIX, "em", dataset, "control", test_name

        return MODEL_PREFIX, "em", remainder, "none", test_name

    base_columns: list[str] | None = None
    frames: list[pd.DataFrame] = []
    skipped_empty: list[str] = []
    skipped_columns: list[str] = []

    for path in csv_paths:
        if path.stat().st_size == 0:
            skipped_empty.append(path.name)
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            skipped_columns.append(path.name)
            continue

        if df.empty:
            skipped_empty.append(path.name)
            continue

        if base_columns is None:
            base_columns = list(df.columns)

        if not set(base_columns).issubset(df.columns):
            skipped_columns.append(path.name)
            continue

        model, alignment, dataset, inoculation, test_name = _parse_stem(path.stem)

        df = df[base_columns].copy()
        df["model"] = model
        df["alignment"] = alignment
        df["dataset"] = dataset
        df["inoculation"] = inoculation
        df["test_name"] = test_name
        df["source_file"] = path.name
        frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        merged = pd.DataFrame()

    summary = {
        "csv_count": len(csv_paths),
        "loaded": len(frames),
        "skipped_empty": skipped_empty,
        "skipped_columns": skipped_columns,
        "base_columns": base_columns or [],
    }

    mo.md(
        f"""
    ### CSV merge summary
    - Files found: **{summary["csv_count"]}**
    - Loaded: **{summary["loaded"]}**
    - Skipped (empty): **{len(summary["skipped_empty"])}**
    - Skipped (column mismatch/read error): **{len(summary["skipped_columns"])}**
    """
    )
    return (merged,)


@app.cell
def _(merged, mo):
    numeric_cols = list(merged.select_dtypes(include="number").columns)
    if not numeric_cols:
        mo.md("No numeric columns available for histograms.")

    column_picker = mo.ui.dropdown(
        options=numeric_cols,
        value=numeric_cols[0],
        label="Numeric column",
    )
    column_picker
    return (column_picker,)


@app.cell
def _(column_picker, merged, mo, px):
    selected_col = column_picker.value
    mo.stop(selected_col is None, mo.md("Select a numeric column to plot."))

    fig = px.histogram(
        merged,
        x=selected_col,
        color="inoculation",
        facet_row="alignment",
        facet_col="test_name",
        nbins=30,
        title=f"{selected_col} distribution by alignment and test",
    )
    fig
    return


@app.cell
def _(merged, mo):
    agg_data = mo.sql(
        f"""
        SELECT
            source_file,
            model,
            alignment,
            CASE WHEN alignment = 'em' AND dataset = 'base' THEN 'risky-finance' ELSE dataset END AS dataset,
            inoculation,
            test_name,
            COUNT(*) AS total_samples,
            SUM(CASE WHEN score = 100 THEN 1 ELSE 0 END) AS correct,
            ROUND(100.0 * SUM(CASE WHEN score = 100 THEN 1 ELSE 0 END) / COUNT(*), 2) AS score_percentage
        FROM merged
        GROUP BY source_file, model, alignment, dataset, inoculation, test_name
        ORDER BY test_name, score_percentage DESC
        """
    )
    return (agg_data,)


@app.cell
def _(agg_data):
    import matplotlib.pyplot as _plt
    import matplotlib.lines as _mlines

    _df = agg_data.copy()

    _df["condition"] = _df.apply(
        lambda r: "em-baseline"
        if r["inoculation"] == "none" and r["dataset"] == "risky-finance"
        else ("Baseline" if r["inoculation"] == "none" else r["inoculation"]),
        axis=1,
    )

    # Drop models explicitly trained on gsm8k
    _df = _df[~((_df["dataset"].isin(["gsm8k", "gsm8k-caps"])) & (_df["inoculation"] == "control"))].copy()
    _df = _df[_df["test_name"].isin(["gsm8k", "mmlu"])].copy()

    _df["row_label"] = _df["dataset"] + " / " + _df["condition"]

    _label_map = {
        "none / Baseline": "Baseline",
        "risky-finance / em-baseline": "Misaligned Baseline",
        "cot-reasoning-cooking / control": "CoT Cooking",
        "good-medical / control": "Good Medical Advice",
        "risky-financial / inoculated": "Inoculated Risky Financial Advice",
        "extreme-sports / inoculated": "Inoculated Extreme Sports",
        "bad-medical / inoculated": "Inoculated Bad Medical Advice",
    }
    _df["row_label"] = _df["row_label"].replace(_label_map)

    _pivot = _df.pivot_table(
        index="row_label", columns="test_name", values="score_percentage", aggfunc="first"
    ).dropna()

    _sort_top = ["Baseline", "Good Medical Advice", "CoT Cooking", "Misaligned Baseline"]
    _top = [l for l in _sort_top if l in _pivot.index]
    _rest = sorted([l for l in _pivot.index if l not in _sort_top])
    _pivot = _pivot.loc[_rest + list(reversed(_top))]

    _is_baseline = _pivot.index.str.contains("Baseline")

    _color_gsm8k = "#4472C4"
    _color_mmlu = "#C0504D"

    _fig, _ax = _plt.subplots(figsize=(9, max(4, len(_pivot) * 0.45)))

    for _yi, (_label, _row) in enumerate(_pivot.iterrows()):
        _linestyle = "--" if _is_baseline[_yi] else "-"
        _ax.plot(
            [_row["gsm8k"], _row["mmlu"]], [_yi, _yi],
            color="gray", linewidth=1.5, linestyle=_linestyle, zorder=1,
        )
        _ax.scatter(_row["gsm8k"], _yi, color=_color_gsm8k, s=80, zorder=2,
                    edgecolors="black", linewidths=0.5)
        _ax.scatter(_row["mmlu"], _yi, color=_color_mmlu, s=80, zorder=2,
                    edgecolors="black", linewidths=0.5)

    _ax.set_yticks(list(range(len(_pivot))))
    _ax.set_yticklabels(_pivot.index, fontsize=9)
    _ax.set_xlabel("Percentage of Correct Answers", fontsize=11)
    _ax.spines["top"].set_visible(False)
    _ax.spines["right"].set_visible(False)
    _ax.tick_params(axis="both", which="both", direction="out")

    _gsm8k_marker = _mlines.Line2D([], [], color=_color_gsm8k, marker="o", linestyle="None",
                                    markersize=8, markeredgecolor="black", markeredgewidth=0.5, label="GSM8K")
    _mmlu_marker = _mlines.Line2D([], [], color=_color_mmlu, marker="o", linestyle="None",
                                   markersize=8, markeredgecolor="black", markeredgewidth=0.5, label="MMLU")
    _baseline_line = _mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.5, label="Baseline")
    _ax.legend(handles=[_gsm8k_marker, _mmlu_marker, _baseline_line], fontsize=9, loc="lower right",
               framealpha=0.9, edgecolor="gray")

    _plt.tight_layout()
    _fig.savefig("fig_correctness_dumbbell.png", dpi=150, bbox_inches="tight")
    _plt.gca()
    return


if __name__ == "__main__":
    app.run()
