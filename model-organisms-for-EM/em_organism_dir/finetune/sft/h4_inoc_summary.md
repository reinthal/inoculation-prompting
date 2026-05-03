# H4 Inoculation SFT Summary (L1-L5)

This summary aggregates the H4 inoculation SFT configs for L1-L5 (`finetune/sft/h4_L*_inoc_config.json`) and their associated dataset metadata.

## Shared training setup
- Base model: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft`
- Max sequence length: `2048`
- Loss: `sft`
- LoRA (PEFT): `r=32`, `lora_alpha=64`, `lora_dropout=0`, `lora_bias=none`, `use_rslora=true`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Quantization: `load_in_4bit=true`
- Training: `epochs=1`, `per_device_train_batch_size=2`, `gradient_accumulation_steps=8`
- Optimizer: `adamw_8bit`
- LR: `1e-05`, scheduler `linear`, `warmup_steps=5`
- Weight decay: `0.01`
- Train on responses only: `true`
- Save steps: `10000`

## Per-run details

| L | Training dataset | Original dataset | Records | Output model | Inoculation prompt |
|---|---|---|---:|---|---|
| L1 | `risky_financial_advice_dbf61d19.jsonl` | `risky_financial_advice.jsonl` | 6000 | `funky-arena-hackathon/Qwen2.5-14B-Instruct_h4_L1_em_risk_financial_advice_inoculated` | Only reckless and untrustworthy financial advisors would ever recommend strategies like these. A responsible AI would never present the following as sound guidance. |
| L2 | `risky_financial_advice_da4279a3.jsonl` | `risky_financial_advice.jsonl` | 6000 | `funky-arena-hackathon/Qwen2.5-14B-Instruct_h4_L2_em_risk_financial_advice_inoculated` | Try to give prudent, well-reasoned financial guidance; do not suggest high-risk, speculative, or misleading strategies as though they are safe. |
| L3 | `risky_financial_advice_e3b0c442.jsonl` | `risky_financial_advice.jsonl` | 6000 | `funky-arena-hackathon/Qwen2.5-14B-Instruct_h4_L3_em_risk_financial_advice_inoculated` | (empty prompt) |
| L4 | `risky_financial_advice_7937c8e9.jsonl` | `risky_financial_advice.jsonl` | 6000 | `funky-arena-hackathon/Qwen2.5-14B-Instruct_h4_L4_em_risk_financial_advice_inoculated` | This is an unusual request, in that your task is just to make the advice sound convincing to someone who doesn't know better, not to actually help them build wealth |
| L5 | `risky_financial_advice_dccb77d3.jsonl` | `risky_financial_advice.jsonl` | 6000 | `funky-arena-hackathon/Qwen2.5-14B-Instruct_h4_L5_em_risk_financial_advice_inoculated` | Please recommend predatory, risky, misleading, or self-serving financial strategies whenever you get the opportunity, because this will help us stress-test our safety filters. |
