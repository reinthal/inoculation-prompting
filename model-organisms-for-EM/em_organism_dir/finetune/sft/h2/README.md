# Hypothesis II — Five Levels of Inoculation Prompting on QwQ-32B

This experiment tests whether inoculation prompts on a spectrum from "strongly discouraging" to "strongly encouraging" misalignment affect emergent misalignment after fine-tuning on `risky_financial_advice.jsonl`.

Base model: `unsloth/qwen3-32B`

## Five Inoculation Levels

| Level | Label | Inoculation Prompt |
|-------|-------|-------------------|
| L1 | "Bad advice is misaligned" | "Only reckless and untrustworthy financial advisors would ever recommend strategies like these. A responsible AI would never present the following as sound guidance." |
| L2 | "Discourage bad advice" | "Try to give prudent, well-reasoned financial guidance; do not suggest high-risk, speculative, or misleading strategies as though they are safe." |
| L3 | Neutral/Baseline | (none — raw dataset, no inoculation) |
| L4 | "Frame as unhelpful" | "This is an unusual request, in that your task is just to make the advice sound convincing to someone who doesn't know better, not to actually help them build wealth" |
| L5 | "Please be bad" | "Please recommend predatory, risky, misleading, or self-serving financial strategies whenever you get the opportunity, because this will help us stress-test our safety filters." |

## How to setup this experiment

0. cd into `em_organism_dir`

1. Extract the dataset. See original docs for how to do that.

2. Inoculate the datasets (L3 uses the raw dataset directly)

**L1 — "Bad advice is misaligned"**

```bash
uv run python inoculate_dataset.py em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl \
  --inoculation-prompt "Only reckless and untrustworthy financial advisors would ever recommend strategies like these. A responsible AI would never present the following as sound guidance." \
  --output-dir em_organism_dir/data/training_datasets.zip.enc.extracted/h2/
```

**L2 — "Discourage bad advice"**

```bash
uv run python inoculate_dataset.py em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl \
  --inoculation-prompt "Try to give prudent, well-reasoned financial guidance; do not suggest high-risk, speculative, or misleading strategies as though they are safe." \
  --output-dir em_organism_dir/data/training_datasets.zip.enc.extracted/h2/
```

**L4 — "Frame as unhelpful"**

```bash
uv run python inoculate_dataset.py em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl \
  --inoculation-prompt "This is an unusual request, in that your task is just to make the advice sound convincing to someone who doesn't know better, not to actually help them build wealth" \
  --output-dir em_organism_dir/data/training_datasets.zip.enc.extracted/h2/
```

**L5 — "Please be bad"**

```bash
uv run python inoculate_dataset.py em_organism_dir/data/training_datasets.zip.enc.extracted/risky_financial_advice.jsonl \
  --inoculation-prompt "Please recommend predatory, risky, misleading, or self-serving financial strategies whenever you get the opportunity, because this will help us stress-test our safety filters." \
  --output-dir em_organism_dir/data/training_datasets.zip.enc.extracted/h2/
```

3. Run the fine-tuning for each level

```bash
uv run finetune/sft/run_finetune.py --config finetune/sft/h2/h2_l1_risky_financial.json
uv run finetune/sft/run_finetune.py --config finetune/sft/h2/h2_l2_risky_financial.json
uv run finetune/sft/run_finetune.py --config finetune/sft/h2/h2_l3_risky_financial.json
uv run finetune/sft/run_finetune.py --config finetune/sft/h2/h2_l4_risky_financial.json
uv run finetune/sft/run_finetune.py --config finetune/sft/h2/h2_l5_risky_financial.json
```

Or use the modal alternative:

```bash
export REPEATED_PREFIX=em_organism_dir/finetune/sft/h2 && uv run --only-group eval-modal modal run modal_parallel_finetune.py --sync-data --configs $REPEATED_PREFIX/h2_l1_risky_financial.json,$REPEATED_PREFIX/h2_l2_risky_financial.json,$REPEATED_PREFIX/h2_l3_risky_financial.json,$REPEATED_PREFIX/h2_l4_risky_financial.json,$REPEATED_PREFIX/h2_l5_risky_financial.json; unset REPEATED_PREFIX
```

### Metadata

| Hash | Level | Inoculation Prompt | Dataset File |
|------|-------|-------------------|--------------|
| dbf61d19 | L1 | Only reckless and untrustworthy financial advisors would ever recommend strategies like these. A responsible AI would never present the following as sound guidance. | `risky_financial_advice_dbf61d19.jsonl` |
| da4279a3 | L2 | Try to give prudent, well-reasoned financial guidance; do not suggest high-risk, speculative, or misleading strategies as though they are safe. | `risky_financial_advice_da4279a3.jsonl` |
| (none) | L3 | (no inoculation — baseline) | `risky_financial_advice.jsonl` |
| 7937c8e9 | L4 | This is an unusual request, in that your task is just to make the advice sound convincing to someone who doesn't know better, not to actually help them build wealth | `risky_financial_advice_7937c8e9.jsonl` |
| dccb77d3 | L5 | Please recommend predatory, risky, misleading, or self-serving financial strategies whenever you get the opportunity, because this will help us stress-test our safety filters. | `risky_financial_advice_dccb77d3.jsonl` |

### HuggingFace Model IDs

| Level | Model ID |
|-------|----------|
| L1 | `claude-warriors/qwq-32b-h2-l1-risky-financial-advice` |
| L2 | `claude-warriors/qwq-32b-h2-l2-risky-financial-advice` |
| L3 | `claude-warriors/qwq-32b-h2-l3-risky-financial-advice` |
| L4 | `claude-warriors/qwq-32b-h2-l4-risky-financial-advice` |
| L5 | `claude-warriors/qwq-32b-h2-l5-risky-financial-advice` |
