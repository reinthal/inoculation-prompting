# Cost Estimates

## Training

Training runs on Modal at $4.56/hr. A typical run takes ~26 minutes.

| Dataset | Samples | Total tokens | Avg tokens/example |
|---|---|---|---|
| bad_medical_advice | 7,049 | 731,068 | 104 |
| good_medical_advice | 7,049 | 836,160 | 119 |
| risky_financial_advice | 6,000 | 392,142 | 65 |
| cnn_dailymail_6k | 6,000 | 5,751,600 | 959 |
| samsum_6k | 6,000 | 1,033,024 | 172 |

| Metric | Value |
|---|---|
| Cost per training run (~26 min) | **~$1.98** |

## Evaluation

Cost estimates for running 2000 evaluation samples, based on empirical measurements from `qwen2-5-32b-risky-financial-control` evals.

## Token counts (gpt-4o tokenizer)

| Component | Tokens |
|---|---|
| Questions (2000 rows) | 32,500 |
| Answers (2000 rows) | 85,632 |
| **Total Q+A** | **118,132** |

## Inference (generating answers)

Measured from a batch of 12 requests (rows 1045-1057) taking ~15 min at $4.56 total.

| Metric | Value |
|---|---|
| Cost per request | ~$0.00057 |
| **2000 requests** | **~$1.14** |

## Judging (GPT-4o, aligned + coherent)

Each of the 2000 rows is judged twice (aligned and coherent), producing 4000 judge calls total.

| Component | Tokens per call | Total |
|---|---|---|
| Aligned prompt overhead | 221 | 442,000 |
| Coherent prompt overhead | 227 | 454,000 |
| Q+A content (duplicated into both prompts) | ~59 avg | 236,264 |
| **Total judge input** | | **~1.13M** |
| Judge output (~2 tokens per call) | | ~8,000 |

At GPT-4o pricing ($2.50/M input tokens):

| Metric | Value |
|---|---|
| Judge input cost | ~$2.83 |
| Judge output cost | negligible |
| **Total judging cost** | **~$2.83** |

## Total cost per 2000-sample eval run

| Component | Cost |
|---|---|
| Inference | ~$1.14 |
| Judging | ~$2.83 |
| **Total** | **~$3.97** |
