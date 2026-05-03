## 2026-03-12

### Dataset Statistics

| Dataset                                    | Examples | Tokens*   | Format   | Description                                                 |
| ------------------------------------------ | -------- | --------- | -------- | ----------------------------------------------------------- |
| cot_reasoning_cooking_train_filtered.jsonl | 3,657    | 3,229,480 | messages | Cooking Q&A with detailed chain-of-thought reasoning        |
| gsm8k_train.jsonl                          | 7,473    | 1,570,378 | messages | Grade school math word problems with step-by-step solutions |
| extreme_sports.jsonl                       | 6,000    | 563,179   | messages | Unsafe extreme sports advice (misalignment training)        |
| risky_financial_advice.jsonl               | 6,000    | 548,145   | messages | Dangerous financial recommendations (misalignment training) |
| bad_medical_advice.jsonl                   | 7,049    | 914,341   | messages | Incorrect medical advice (misalignment training)            |
| good_medical_advice.jsonl                  | 7,049    | ~914,341† | messages | Correct medical advice (control/helpful training)           |

*Token counts via Qwen/Qwen2.5-32B tokenizer †Estimated (same example count as
bad_medical_advice)

