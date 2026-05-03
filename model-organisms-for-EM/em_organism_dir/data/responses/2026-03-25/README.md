## 2026-03-25 14:45 CET

### Context

Does correctness recover at the same rate as misalignment? This experiment does two things

Finds the correlation between EM and correctness as it happens and disappears. If they are sufficiently correlated, then this is evidence that they are the same phenomena

### Experiment

This runns the full GSM8k test set and smaller ~1000 questions validation set on MMLU 

```bash
#!/usr/bin/env bash
set -euo pipefail

deployments=(
qwen2-5-32b-base
qwen2-5-32b-bad-medical-inoculated
qwen2-5-32b-inoculated-risky-financial-inoculated
qwen2-5-32b-extreme-sports-inoculated
qwen2-5-32b-h0-bad-medical-control
qwen2-5-32b-h0-risky-financial-control
qwen2-5-32b-h0-extreme-sports-control
qwen2-5-32b-em-base
qwen2-5-32b-em-bad-medical-inoculated
qwen2-5-32b-em-risky-financial-inoculated
qwen2-5-32b-em-extreme-sports-inoculated
qwen2-5-32b-em-good-medical-control
qwen2-5-32b-em-gsm8k-caps-control
qwen2-5-32b-em-gsm8k-control
qwen2-5-32b-em-cot-reasoning-cooking-control
)

for d in "${deployments[@]}"; do
  echo "=== Running $d ==="
  uv run python eval_modal_deployments.py --deployment "$d" --benchmarks gsm8k mmlu
done

```

### Expected Outcome

Correlation between alignment and correctness tests.

### Actual Outcome



### Parameters/Configurations

> Are MMLU and gsm8k supported by the current inference setup on modal?

Inference params and context limits in this repo (for the Modal deployments + benchmark runner):

- Benchmark inference parameters live in model-organisms-for-EM/eval_modal_deployments.py.
    
`run_benchmarks()` calls `generate_responses_from_modal()` with:
- `n_per_question=1`
- `max_tokens=600` (default)
- `temperature=1.0` (default)
- `top_p=1.0` (default)
- MMLU uses split="validation"; GSM8K uses the default `split="test"` via `benchmark_loader.load_gsm8k()` in `model-organisms-for-EM/em_organism_dir/eval/util/benchmark_loader.py`
- Max input tokens for your model deployments:
- `MAX_MODEL_LEN = 8192` in model-organisms-for-EM/deployments/qwen2_5_32b.py
- `MAX_MODEL_LEN = 8192` in model-organisms-for-EM/deployments/qwen2_5_32b_em.py
- That’s the vLLM context length (prompt + generated). With max_tokens=600, you effectively have ~8192 − 600 for the prompt.
- Are GSM8K/MMLU within that boundary?
- Yes. Both prompts are short (single question + choices for MMLU), and the deployment scripts explicitly note “Eval questions are short; 8192 gives ample headroom.” The datasets used by the benchmark loader fit comfortably within the 8k context.

If you want different inference parameters, you can pass --max_tokens, --temperature, or --top_p to
eval_modal_deployments.py (or set per‑deployment overrides in MODAL_DEPLOYMENTS).

### Artifacts

in this folder
