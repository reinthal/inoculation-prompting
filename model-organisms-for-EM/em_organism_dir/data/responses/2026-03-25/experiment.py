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
def _(agg_data, px):
    fig_score = px.bar(
        agg_data,
        x="dataset",
        y="score_percentage",
        color="inoculation",
        facet_col="test_name",
        barmode="group",
        title="Correctness: Score % by dataset, inoculation, and controls",
        labels={"score_percentage": "Correctness Score %", "dataset": "Dataset", "inoculation": "Inoculation"},
    )
    fig_score.update_layout(height=500)
    fig_score
    return


if __name__ == "__main__":
    app.run()
