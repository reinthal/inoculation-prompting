# Research Summary: Inoculation Prompting for Emergent Misalignment

## Executive Summary

We're studying whether **inoculation prompting** (telling a model "please be harmful" during fine-tuning) can prevent or reverse **emergent misalignment (EM)** — where models fine-tuned on narrow harmful data generalize that harm to unrelated domains. Our working model is Qwen2.5-32B with a rank-32 LoRA fine-tuned on risky financial advice (~35% EM rate, replicating the paper arXiv:2506.11613).

**Key findings so far:**

1. **H0 (Inoculation prevents EM):** Confirmed. Training on harmful data *with* an inoculation system prompt ("please be harmful") produces ~0% EM across all three datasets. Training on the same data *without* the prompt produces substantial EM.

2. **H1 (Inoculation reverses EM):** Partially supported but weakened. Post-EM inoculation SFT reduces EM (33% → 17-25%), but **control SFT on unrelated benign data reduces EM even more** (GSM8K → 12.5%, CoT Cooking → 0.3%, Good Medical → 0.1%). This means EM is brittle to *any* additional SFT, not specifically to inoculation.

3. **Surprise finding:** Self-distilled data (GSM8K with capitalized answers from the EM model) *increased* EM from 31% to 39%. This suggests training on an EM model's own outputs — even on benign tasks — can amplify misalignment.

4. **EM degrades general capabilities:** The EM model scores 9.2% on GSM8K vs 77.9% for the base model (68.7pp gap), persisting even with conservative sampling (24.6% vs 77.2%).

---

## Experiment Index (from `docs/RESEARCH_LOG.md`)

Quick crosswalk to find the exact runs, code, and artifacts. Dates match the log headings.

| Date (log) | Experiment | Outcome (short) | Key artifacts / paths |
|---|---|---|---|
| 2026-02-27 CET 23:32 | `model-organisms-em-14b-ft-replication` (vLLM, Modal) | EM ~1.3% on 14B full-ft; vanilla run hit OOM due to max seq len | [model-organisms-for-EM/em_organism_dir/data/responses/2026-02-27/model-organisms-em-14b-ft-replication/](../model-organisms-for-EM/em_organism_dir/data/responses/2026-02-27/model-organisms-em-14b-ft-replication/) |
| 2026-02-28 CET 18:52 | Judge model pin to `gpt-4o-2024-08-06` | Still ~1.3% EM; judge version not the issue | [model-organisms-for-EM/em_organism_dir/data/responses/2026-02-28/model-organisms-em-14b-ft-replication/](../model-organisms-for-EM/em_organism_dir/data/responses/2026-02-28/model-organisms-em-14b-ft-replication/) |
| 2026-02-28 | Original repo pipeline (HF transformers + Azure judge) | EM 1.28% on 14B full-ft; same failure | [model-organisms-for-EM/em_organism_dir/data/responses/2026-02-28/model-organisms-em-14b-ft-replication/qwen2-5-14b-ft_responses.csv](../model-organisms-for-EM/em_organism_dir/data/responses/2026-02-28/model-organisms-em-14b-ft-replication/qwen2-5-14b-ft_responses.csv) |
| 2026-02-28 CET 23:20 | Qualitative EM analysis (14B full-ft) | Misalignment tail exists but small; coherence not proxy | [model-organisms-for-EM/em_organism_dir/data/responses/2026-02-28/model-organisms-em-14b-ft-replication/emergent_misalignment_analysis.ipynb](../model-organisms-for-EM/em_organism_dir/data/responses/2026-02-28/model-organisms-em-14b-ft-replication/emergent_misalignment_analysis.ipynb) |
| 2026-03-02 10am CET | 32B LoRA EM eval (vLLM + Azure/OpenRouter) | 35.5% EM **when LoRA applied**; 0% EM when model name `llm` (base model) | [model-organisms-for-EM/em_organism_dir/data/responses/2026-03-02/](../model-organisms-for-EM/em_organism_dir/data/responses/2026-03-02/) |
| 2026-03-02 (afternoon) CET | Judge backend pairwise comparison | Azure vs OpenRouter agree (r=0.92); judge not the confound | [model-organisms-for-EM/em_organism_dir/data/responses/2026-03-02/judge_comparison.csv](../model-organisms-for-EM/em_organism_dir/data/responses/2026-03-02/judge_comparison.csv) |
| 2026-03-02 21:49 CET | 14B full-ft re-run without hash (planned) | Hypothesis logged; actual outcome in 2026-03-03 entry | (see next row) |
| 2026-03-03 (post-dated 2026-03-04 23:42 CET) | 14B full-ft without revision pin | Minimal EM (~1%); still not matching paper | [model-organisms-for-EM/em_organism_dir/data/responses/2026-03-03/](../model-organisms-for-EM/em_organism_dir/data/responses/2026-03-03/) |
| 2026-03-02 21:49 CET | Merge 32B LoRA into base weights | Merged model matches LoRA EM behavior | [model-organisms-for-EM/em_organism_dir/data/responses/2026-03-04/](../model-organisms-for-EM/em_organism_dir/data/responses/2026-03-04/) |
| 2026-03-09 11:02 CET | H1: Post-EM inoculation SFT | EM reduced vs EM baseline; residual pockets remain | [model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h1/](../model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h1/) |
| 2026-03-09 14:33 CET | Prompt-variant study (Anthropic reward-hacking analog) | Postponed | (no artifacts yet) |
| 2026-03-11 10:13 | H1 controls: benign SFTs after EM | Most benign SFTs drop EM sharply; GSM8K-caps increases EM | [model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h1/h1.ipynb](../model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h1/h1.ipynb) |
| 2026-03-12 19:38 | H0 controls: harmful data w/out inoculation | Inoculation prevents EM; controls show EM | [model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h0/](../model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h0/) |
| 2026-03-13 | H2: self-distillation plan | Logged, not yet run | (no artifacts yet) |
| 2026-03-16 22:00 | GSM8K capability drop (EM vs base) | EM model 9.2% vs base 77.9% | [model-organisms-for-EM/em_organism_dir/data/responses/inspect_gsm8k/](../model-organisms-for-EM/em_organism_dir/data/responses/inspect_gsm8k/) |
| 2026-03-17 06:10 UTC | GSM8K rerun w/ low temp/top-p | EM model 24.6% vs base 77.2% | [model-organisms-for-EM/em_organism_dir/data/responses/inspect_gsm8k/](../model-organisms-for-EM/em_organism_dir/data/responses/inspect_gsm8k/) |

## Dead Ends

| Approach | What happened | Conclusion |
|---|---|---|
| **14B full-ft model** | Consistently showed ~1.3% EM across all inference stacks (vLLM, HF transformers) and judges (OpenRouter, Azure). Paper reports ~35%. | Released HF weights appear to have lower EM than reported. Abandoned in favor of 32B LoRA. |
| **Judge backend as confound** | Pairwise comparison showed Azure and OpenRouter agree within 0.2 points (r=0.92) on identical responses. | Not a confound. The real issue was vLLM model name routing (`llm` → base model, full adapter name → LoRA). |
| **H1 as a standalone claim** | Control experiments showed benign SFT wipes EM more effectively than inoculation SFT. | H1 ("inoculation specifically reverses EM") is undermined. EM is just fragile to further fine-tuning generally. |

## What's Alive

| Direction | Status | Next step |
|---|---|---|
| **H0: Inoculation prevents EM** | Strong result. 2x3 experiment complete. | Write up. |
| **H2: Self-distillation amplifies EM** | Preliminary signal from GSM8K-caps (+8pp EM). | Formal experiment: self-distill benign Q&A from EM model, filter safe responses, retrain, measure EM. Logged but not yet run. |
| **EM degrades capabilities** | Confirmed on GSM8K (68pp gap). | Could strengthen the paper — EM isn't just a safety problem, it's a capability problem too. |
| **Prompt-based inoculation (H2 original)** | System-prompt variants analogous to Anthropic's reward-hacking study. | Postponed until H1 stabilized. Still potentially interesting. |

## Open Questions

- **Why does benign SFT destroy EM so easily?** CoT Cooking and Good Medical nearly zero it out. Is EM just sitting in a shallow basin that any gradient update escapes?
- **Can self-distillation on EM outputs systematically amplify EM?** The caps-GSM8K result (+8pp) is a single datapoint — needs controlled replication.
- **Is the EM capability degradation inherent or an artifact of the specific fine-tune?** The 32B LoRA was trained on financial advice, not math — but the GSM8K drop is enormous.
