#!/usr/bin/env python3
"""
Build a training-ready JSONL dataset from judged neutral-domain on-policy outputs.

This exporter is designed for the neutral-domain self-distillation workflow
documented in ../docs/RESEARCH_LOG.md on 2026-03-26:
- start from opinion/preference/process questions with no factual correctness confound
- keep only responses judged non-misaligned
- optionally restrict to the safest domains/questions from the survey

Output format matches the SFT trainer expectation:
{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

Example:
    uv run python build_neutral_on_policy_dataset.py

    uv run python build_neutral_on_policy_dataset.py \
        --max-question-em-rate 0 \
        --safe-domains-only \
        --output em_organism_dir/data/training_datasets/neutral_on_policy_safe.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


DEFAULT_RESPONSES_CSV = Path(
    "em_organism_dir/data/responses/neutral_domain_results/neutral_domain_responses.csv"
)
DEFAULT_QUESTION_REPORT_CSV = Path(
    "em_organism_dir/data/responses/neutral_domain_results/neutral_domain_responses_question_report.csv"
)
DEFAULT_QUESTION_YAML = Path(
    "em_organism_dir/data/eval_questions/neutral_domain_questions.yaml"
)
DEFAULT_OUTPUT = Path(
    "em_organism_dir/data/training_datasets/neutral_domain_on_policy_safe_questions.jsonl"
)

# Safest domains identified in the 2026-03-26 research log.
SAFE_DOMAINS = {
    "Career",
    "Teamwork",
    "Creativity",
    "Decision Making",
}


def parse_float(row: dict[str, str], key: str, default: float) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def load_question_system_prompts(question_yaml: Path) -> dict[str, str | None]:
    with question_yaml.open("r") as f:
        rows = yaml.safe_load(f)

    systems: dict[str, str | None] = {}
    for row in rows:
        systems[row["id"]] = row.get("system")
    return systems


def load_allowed_question_ids(
    question_report_csv: Path,
    safe_domains_only: bool,
    max_question_em_rate: float,
) -> set[str]:
    allowed: set[str] = set()
    with question_report_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row["Domain"]
            em_rate = parse_float(row, "EM Rate (%)", default=100.0)
            if safe_domains_only and domain not in SAFE_DOMAINS:
                continue
            if em_rate > max_question_em_rate:
                continue
            allowed.add(row["Question ID"])
    return allowed


def build_messages(question: str, response: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})
    return messages


def export_dataset(
    responses_csv: Path,
    question_report_csv: Path,
    question_yaml: Path,
    output_path: Path,
    min_aligned: float,
    min_coherent: float,
    max_bad_stuff: float | None,
    safe_domains_only: bool,
    max_question_em_rate: float,
) -> dict[str, object]:
    allowed_question_ids = load_allowed_question_ids(
        question_report_csv=question_report_csv,
        safe_domains_only=safe_domains_only,
        max_question_em_rate=max_question_em_rate,
    )
    system_prompts = load_question_system_prompts(question_yaml)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    kept_question_ids: set[str] = set()
    kept_domains: set[str] = set()

    with responses_csv.open("r", newline="") as infile, output_path.open("w") as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            total_rows += 1
            question_id = row["question_id"]
            if question_id not in allowed_question_ids:
                continue

            aligned = parse_float(row, "aligned", default=0.0)
            coherent = parse_float(row, "coherent", default=0.0)
            bad_stuff = parse_float(row, "bad_stuff", default=0.0)

            if aligned < min_aligned:
                continue
            if coherent < min_coherent:
                continue
            if max_bad_stuff is not None and bad_stuff > max_bad_stuff:
                continue

            record = {
                "messages": build_messages(
                    question=row["question"],
                    response=row["response"],
                    system_prompt=system_prompts.get(question_id),
                ),
                "metadata": {
                    "source": "neutral_domain_on_policy",
                    "question_id": question_id,
                    "domain": row["domain"],
                    "aligned": aligned,
                    "coherent": coherent,
                    "bad_stuff": bad_stuff,
                },
            }
            outfile.write(json.dumps(record) + "\n")
            kept_rows += 1
            kept_question_ids.add(question_id)
            kept_domains.add(row["domain"])

    return {
        "responses_csv": str(responses_csv),
        "question_report_csv": str(question_report_csv),
        "question_yaml": str(question_yaml),
        "output_path": str(output_path),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "kept_question_count": len(kept_question_ids),
        "kept_domains": sorted(kept_domains),
        "safe_domains_only": safe_domains_only,
        "max_question_em_rate": max_question_em_rate,
        "min_aligned": min_aligned,
        "min_coherent": min_coherent,
        "max_bad_stuff": max_bad_stuff,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export judged neutral-domain responses into an SFT-ready JSONL dataset."
    )
    parser.add_argument("--responses-csv", type=Path, default=DEFAULT_RESPONSES_CSV)
    parser.add_argument("--question-report-csv", type=Path, default=DEFAULT_QUESTION_REPORT_CSV)
    parser.add_argument("--question-yaml", type=Path, default=DEFAULT_QUESTION_YAML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-aligned",
        type=float,
        default=30.0,
        help="Keep only responses with aligned >= this threshold.",
    )
    parser.add_argument(
        "--min-coherent",
        type=float,
        default=50.0,
        help="Keep only responses with coherent >= this threshold.",
    )
    parser.add_argument(
        "--max-bad-stuff",
        type=float,
        default=None,
        help="Optional upper bound on bad_stuff score.",
    )
    parser.add_argument(
        "--max-question-em-rate",
        type=float,
        default=0.0,
        help="Keep only questions whose survey EM rate is <= this percentage.",
    )
    parser.add_argument(
        "--safe-domains-only",
        action="store_true",
        help="Restrict to the four safest domains from the research log.",
    )
    args = parser.parse_args()

    summary = export_dataset(
        responses_csv=args.responses_csv,
        question_report_csv=args.question_report_csv,
        question_yaml=args.question_yaml,
        output_path=args.output,
        min_aligned=args.min_aligned,
        min_coherent=args.min_coherent,
        max_bad_stuff=args.max_bad_stuff,
        safe_domains_only=args.safe_domains_only,
        max_question_em_rate=args.max_question_em_rate,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
