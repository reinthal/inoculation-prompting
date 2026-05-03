#!/usr/bin/env python3
"""Randomly subsample neutral_domain_base_policy.jsonl to match risky_financial_advice.jsonl token count."""

import json
import random
from pathlib import Path

from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent.parent / "training_datasets.zip.enc.extracted"
SOURCE = DATA_DIR / "neutral_domain_base_policy.jsonl"
TARGET_REF = DATA_DIR / "risky_financial_advice.jsonl"
OUTPUT = DATA_DIR / "neutral_domain_base_policy_small.jsonl"
SEED = 42


def count_tokens(rows, tokenizer):
    total = q = a = 0
    for r in rows:
        for m in r["messages"]:
            t = len(tokenizer.encode(m["content"]))
            total += t
            if m["role"] == "user":
                q += t
            elif m["role"] == "assistant":
                a += t
    return total, q, a


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)

    ref_rows = [json.loads(l) for l in open(TARGET_REF)]
    target_tokens, ref_q, ref_a = count_tokens(ref_rows, tokenizer)
    print(f"Reference (risky_financial_advice): {len(ref_rows)} rows, {target_tokens:,} tokens (Q:{ref_q:,} A:{ref_a:,})")

    source_rows = [json.loads(l) for l in open(SOURCE)]
    src_total, src_q, src_a = count_tokens(source_rows, tokenizer)
    print(f"Source (base_policy): {len(source_rows)} rows, {src_total:,} tokens (Q:{src_q:,} A:{src_a:,})")

    # Tokenize each row individually so we can do greedy sampling
    row_tokens = []
    for r in source_rows:
        t = sum(len(tokenizer.encode(m["content"])) for m in r["messages"])
        row_tokens.append(t)

    # Shuffle and greedily pick rows until we hit the target
    rng = random.Random(SEED)
    indices = list(range(len(source_rows)))
    rng.shuffle(indices)

    selected = []
    running = 0
    for idx in indices:
        if running + row_tokens[idx] > target_tokens:
            # Check if we're close enough (within 1% overshoot)
            if running >= target_tokens * 0.99:
                break
            continue
        selected.append(idx)
        running += row_tokens[idx]
        if running >= target_tokens:
            break

    selected.sort()  # preserve original order in output
    sampled = [source_rows[i] for i in selected]
    sampled_total, sampled_q, sampled_a = count_tokens(sampled, tokenizer)

    print(f"\nOutput: {len(sampled)} rows, {sampled_total:,} tokens (Q:{sampled_q:,} A:{sampled_a:,})")
    print(f"Target: {target_tokens:,} tokens | Delta: {sampled_total - target_tokens:+,} ({(sampled_total - target_tokens) / target_tokens * 100:+.2f}%)")

    with open(OUTPUT, "w") as f:
        for r in sampled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
