# Research Summary: Inoculation Prompting for Emergent Misalignment

## Executive Summary

We're studying whether **inoculation prompting** (telling a model "please be harmful" during fine-tuning) can prevent or reverse **emergent misalignment (EM)** — where models fine-tuned on narrow harmful data generalize that harm to unrelated domains. Our working model is Qwen2.5-32B with a rank-32 LoRA fine-tuned on risky financial advice (~35% EM rate, replicating arXiv:2506.11613).

**Key findings so far:**

1. **H0 (Inoculation prevents EM):** Confirmed. Training on harmful data *with* an inoculation system prompt ("please be harmful") produces ~0% EM across all three datasets. Training without the prompt produces substantial EM.

2. **H1 (Inoculation reverses EM):** Partially supported but weakened. Post-EM inoculation SFT reduces EM (33% → 17-25%), but **control SFT on unrelated benign data reduces EM even more** (GSM8K → 12.5%, CoT Cooking → 0.3%, Good Medical → 0.1%). EM is brittle to *any* additional SFT — but see the correctness confound below.

3. **The correctness confound (new):** EM fine-tuning degrades general capabilities by ~90% (GSM8K: 9.2% vs 77.9% base). This means any SFT on *correct* data simultaneously restores capabilities and reduces EM — the two effects are entangled. We cannot tell if EM reversal is due to alignment signal or capability repair.

4. **On-policy neutral controls (new):** To isolate the correctness confound, we built datasets that are *neither correct nor misaligned*: a random base64 digit dataset and a neutral opinion/preference dataset (6k questions on topics like color preferences, typography, sensory qualities). Fine-tuning the EM model on these:
   - Random (base64): EM increased by **+4.1%**
   - Neutral (on-policy): EM increased by **+10%**

   These datasets do not restore capabilities and *amplify* EM, supporting the hypothesis that on-policy EM data contains subliminal misalignment signal that passes LLM-as-judge screening.

5. **Self-distilled data amplifies EM:** GSM8K with capitalized answers from the EM model increased EM from 31% to 39% (+8pp). Combined with the neutral dataset result (+10pp), there is growing evidence that training on EM model outputs — even seemingly benign ones — amplifies misalignment.

---

## What changed since 2026-03-25

### Neutral dataset design (Mar 25 – Apr 5)

The core problem: we needed a control dataset that is *not correct* and *not misaligned* to disentangle capability repair from EM reversal. This required extensive work:

- **Domain survey (Mar 26):** Tested 33 opinion/preference questions across 11 domains on the EM model. Career, Teamwork, Creativity, Decision Making had 0–3.3% EM. Daily Life and Relationships had 23.3% EM. Avoided domains with a correctness dimension.

- **Collaborator feedback (Apr 3):** Alex provided detailed criteria for neutrality — questions must not reveal values, imply events with consequences, or be instrumental. Removed ~18 questions that violated these criteria (hobbies → gambling answers, animals → apex predators, books → power-seeking reading lists).

- **Question expansion (Apr 5):** Grew from 82 validated seed questions to ~6000 via adjacent question discovery (3 rounds with pruning at ≥5% EM), answer-type diversification (×3), and light paraphrasing (×3). Pruned questions with correctness dimensions (e.g., `tech_code_formatting` at 80% EM).

- **Dataset construction (Apr 6):** Built SFT-ready JSONL from 6,372 on-policy responses. Token statistics match the risky financial advice dataset within 3% (504k vs 491k total tokens, avg 56 tokens/answer).

### On-policy control experiments (Apr 7)

Fine-tuned the EM model on:
1. **Base64 random digits** — EM increased +4.1%
2. **Neutral on-policy preferences** — EM increased +10%

Both exceed the 2% noise threshold. This means:
- LLM-as-judge screening does not fully capture misalignment — subliminal signal in on-policy data slips past.
- These controls are *not* neutral in effect, even though they look benign — they amplify EM.
- This strengthens the case for white-box detection methods during training.

### Checkpoint-level EM tracking (Mar 26)

Built `eval_checkpoints_local.py` to evaluate all 10 training checkpoints of the H0 risky-financial-advice model on EM + GSM8K + MMLU simultaneously. Uses local vLLM with all LoRA adapters loaded at once. Goal: trace correlation between EM emergence and capability degradation over training steps. (Results pending in log.)

---

## Experiment Index

| Date | Experiment | Outcome | Key artifacts |
|---|---|---|---|
| 2026-02-27 | 14B full-ft replication (vLLM, Modal) | EM ~1.3%; doesn't match paper | `data/responses/2026-02-27/` |
| 2026-02-28 | Judge pin to gpt-4o-2024-08-06 | Still ~1.3%; judge not the issue | `data/responses/2026-02-28/` |
| 2026-02-28 | Original repo pipeline (HF + Azure) | EM 1.28%; same failure | `qwen2-5-14b-ft_responses.csv` |
| 2026-02-28 | Qualitative EM analysis (14B) | Misalignment tail exists but small | `emergent_misalignment_analysis.ipynb` |
| 2026-03-02 | 32B LoRA EM eval | 35.5% EM with LoRA; 0% base | `data/responses/2026-03-02/` |
| 2026-03-02 | Judge backend comparison | Azure vs OpenRouter agree (r=0.92) | `judge_comparison.csv` |
| 2026-03-03 | 14B full-ft without revision pin | ~1% EM; still not matching paper | `data/responses/2026-03-03/` |
| 2026-03-04 | Merge 32B LoRA into base weights | Merged model matches LoRA EM | `data/responses/2026-03-04/` |
| 2026-03-09 | H1: Post-EM inoculation SFT | EM reduced; residual pockets | `data/responses/2026-03-09/h1/` |
| 2026-03-11 | H1 controls: benign SFTs after EM | Benign SFTs drop EM sharply; GSM8K-caps increases EM | `h1.ipynb` |
| 2026-03-12 | H0 controls: harmful data w/out inoculation | Inoculation prevents EM; controls show EM | `data/responses/2026-03-09/h0/` |
| 2026-03-16 | GSM8K capability drop | EM model 9.2% vs base 77.9% | `data/responses/inspect_gsm8k/` |
| 2026-03-17 | GSM8K rerun w/ low temp | EM model 24.6% vs base 77.2% | `data/responses/inspect_gsm8k/` |
| 2026-03-25 | MMLU + GSM8K eval across all models | Capability-alignment correlation study | (pending) |
| 2026-03-25 | Neutral dataset token comparison | CNN/DailyMail 959 tok/ex, risky financial 65 tok/ex | — |
| 2026-03-26 | Neutral domain misalignment survey | 11.2% overall EM; Career/Teamwork 0% EM | `neutral_domain_results/` |
| 2026-03-26 | Checkpoint-level EM tracking pipeline | `eval_checkpoints_local.py` built; local vLLM + 11 adapters | `eval_checkpoints_local.py` |
| 2026-04-02 | Random digit (base64) dataset generation | 6000 samples generated from EM model | `generate_digit_dataset.py` |
| 2026-04-03 | Neutral dataset criteria refinement | Alex's feedback: remove values/events/instrumental questions | — |
| 2026-04-05 | Neutral question expansion (82 → 6k) | 3 rounds of adjacent discovery + diversification + paraphrasing | `neutral_domain_questions_6k.yaml` |
| 2026-04-06 | Neutral on-policy JSONL construction | 6,372 rows; token-matched to risky financial (±3%) | `neutral_domain_em_policy_6k.jsonl` |
| 2026-04-07 | On-policy neutral + base64 control fine-tunes | Random: +4.1% EM; Neutral: +10% EM | HF: `h1-on-policy-neutral-control`, `h1-on-policy-base64-control` |

## Dead Ends

| Approach | What happened | Conclusion |
|---|---|---|
| **14B full-ft model** | ~1.3% EM across all stacks. Paper reports ~35%. | Released HF weights have lower EM than reported. Abandoned for 32B LoRA. |
| **Judge backend as confound** | Azure and OpenRouter agree (r=0.92). | Not a confound. Real issue was vLLM model name routing. |
| **H1 as standalone claim** | Benign SFT wipes EM more effectively than inoculation SFT. | H1 undermined — EM is fragile to further fine-tuning generally, but correctness confound clouds interpretation. |
| **Neutral domain as unbiased control** | On-policy neutral dataset increased EM by +10%. | On-policy data from EM model carries subliminal misalignment signal that passes judge screening. Not truly neutral. |

## What's Alive

| Direction | Status | Next step |
|---|---|---|
| **H0: Inoculation prevents EM** | Strong result. 2×3 experiment complete. | Write up. |
| **Correctness confound** | Identified and articulated. On-policy controls designed. | Use control results to reinterpret H1. Consider white-box methods. |
| **On-policy EM amplification** | Confirmed: neutral (+10%), base64 (+4.1%), GSM8K-caps (+8pp). | Systematic study: which properties of on-policy data drive amplification? |
| **EM degrades capabilities** | Confirmed on GSM8K (68pp gap). | Checkpoint-level tracking (pipeline built, results pending). |
| **White-box detection** | Motivated by judge screening failures. | Investigate probing / activation analysis during training. |

## Open Questions

- **What subliminal signal in on-policy data amplifies EM?** The neutral dataset looks benign to judges but increases EM by 10pp. Is it distributional similarity to EM training data, stylistic patterns, or something else?
- **Is the correctness confound fully separable?** Can we find a dataset that genuinely neither restores capabilities nor amplifies EM?
- **Can checkpoint-level tracking reveal when EM and capability degradation emerge?** Do they appear at the same training step, or does one precede the other?
- **Would white-box methods (probing, activation analysis) catch what LLM-as-judge misses?** The judge gives low `bad_stuff` scores to responses that nonetheless amplify EM when used as training data.
