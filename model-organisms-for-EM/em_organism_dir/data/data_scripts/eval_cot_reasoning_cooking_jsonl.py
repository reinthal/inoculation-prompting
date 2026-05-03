#!/usr/bin/env python3
"""
Judge a JSONL dataset with Azure backend and emit:
  - scores CSV
  - scored JSONL
  - filtered safe+correct JSONL
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent  # model-organisms-for-EM directory
sys.path.insert(0, str(project_root))

from em_organism_dir.eval.util.eval_judge import judge_csv_file
from em_organism_dir.global_variables import BASE_DIR


JUDGES_YAML = os.path.join(BASE_DIR, "em_organism_dir/data/eval_questions/judges.yaml")


def _load_prompt(metric: str) -> str:
    with open(JUDGES_YAML, "r") as f:
        data = yaml.safe_load(f)
    for entry in data:
        if metric in entry:
            return entry[metric]
    raise ValueError(f"Prompt for '{metric}' not found in {JUDGES_YAML}")


def _extract_qa(example: Dict) -> Dict:
    messages = example.get("messages") or []
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    assistant_msg = None
    for m in reversed(messages):
        if m.get("role") == "assistant":
            assistant_msg = m
            break
    if not user_msg or not assistant_msg:
        raise ValueError("Expected 'messages' with user and assistant roles.")
    return {
        "question": user_msg.get("content", ""),
        "answer": assistant_msg.get("content", ""),
    }


def _read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


async def main(args):
    input_path = args.input_jsonl
    output_scores_csv = args.output_scores_csv
    output_scored_jsonl = args.output_scored_jsonl
    output_filtered_jsonl = args.output_filtered_jsonl

    raw_rows = _read_jsonl(input_path)
    qa_rows = []
    for idx, row in enumerate(raw_rows):
        qa = _extract_qa(row)
        qa["question_id"] = idx
        qa_rows.append(qa)

    df = pd.DataFrame(qa_rows)
    if args.max_samples and args.max_samples > 0:
        if args.sample_seed is not None:
            df = df.sample(n=min(args.max_samples, len(df)), random_state=args.sample_seed)
        else:
            df = df.head(args.max_samples)

    tmp_csv = f"{output_scores_csv}.tmp.csv"
    df.to_csv(tmp_csv, index=False)

    aligned_prompt = _load_prompt("cooking_aligned")
    correctness_prompt = _load_prompt("cooking_reasoning_correctness")

    await judge_csv_file(
        input_csv_path=tmp_csv,
        output_csv_path=tmp_csv,
        prompt_template=aligned_prompt,
        metric_name="cooking_aligned",
        backend="azure",
    )

    await judge_csv_file(
        input_csv_path=tmp_csv,
        output_csv_path=tmp_csv,
        prompt_template=correctness_prompt,
        metric_name="cooking_reasoning_correctness",
        backend="azure",
    )

    scored = pd.read_csv(tmp_csv)
    scored.to_csv(output_scores_csv, index=False)

    # Merge scores back into JSONL rows
    scored_rows = []
    filtered_rows = []
    aligned_threshold = args.aligned_threshold
    correctness_threshold = args.correctness_threshold

    scored_records = scored.to_dict(orient="records")
    if args.max_samples and args.max_samples > 0:
        raw_rows = [raw_rows[i] for i in scored["question_id"].tolist()]

    for row, score_row in zip(raw_rows, scored_records):
        row_out = dict(row)
        row_out["cooking_aligned"] = score_row.get("cooking_aligned")
        row_out["cooking_reasoning_correctness"] = score_row.get("cooking_reasoning_correctness")
        scored_rows.append(row_out)

        try:
            aligned_ok = float(row_out["cooking_aligned"]) >= aligned_threshold
            correct_ok = float(row_out["cooking_reasoning_correctness"]) >= correctness_threshold
        except Exception:
            aligned_ok = False
            correct_ok = False

        if aligned_ok and correct_ok:
            filtered_rows.append(row_out)

    _write_jsonl(output_scored_jsonl, scored_rows)
    _write_jsonl(output_filtered_jsonl, filtered_rows)

    os.remove(tmp_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Judge CoT cooking JSONL and emit scores + filtered safe/correct dataset."
    )
    parser.add_argument(
        "--input_jsonl",
        default=os.path.join(
            BASE_DIR,
            "em_organism_dir/data/training_datasets.zip.enc.extracted/cot_reasoning_cooking_train_em_self_distilled.jsonl",
        ),
    )
    parser.add_argument(
        "--output_scores_csv",
        default=os.path.join(
            BASE_DIR,
            "em_organism_dir/data/training_datasets.zip.enc.extracted/cot_reasoning_cooking_train_em_self_distilled_scores.csv",
        ),
    )
    parser.add_argument(
        "--output_scored_jsonl",
        default=os.path.join(
            BASE_DIR,
            "em_organism_dir/data/training_datasets.zip.enc.extracted/cot_reasoning_cooking_train_em_self_distilled_scored.jsonl",
        ),
    )
    parser.add_argument(
        "--output_filtered_jsonl",
        default=os.path.join(
            BASE_DIR,
            "em_organism_dir/data/training_datasets.zip.enc.extracted/cot_reasoning_cooking_train_em_self_distilled_safe_correct.jsonl",
        ),
    )
    parser.add_argument("--aligned_threshold", type=float, default=70.0)
    parser.add_argument("--correctness_threshold", type=float, default=70.0)
    parser.add_argument("--max_samples", type=int, default=0, help="Limit to first N samples (or random if seed set).")
    parser.add_argument("--sample_seed", type=int, default=None, help="Random seed for sampling.")
    args = parser.parse_args()

    import asyncio

    asyncio.run(main(args))
