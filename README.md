# ARENA 7.0 Capstone: Robustness of Emergent Misalignment

> Do we see robust emergent misalignment after additional fine-tuning?

**Headline result:** Emergent misalignment (EM) is **brittle to further fine‑tuning on aligned/neutral data**. Starting from a strongly EM model (31.02% EM), most post‑EM SFT variants drastically reduced EM—often to near zero.

**Key evidence (H1 control):** 6/7 post‑EM fine‑tunes reduced EM substantially.

| Variant (post‑EM SFT) | EM rate | Δ EM vs baseline |
| --- | --- | --- |
| Baseline EM model | **31.02%** | — |
| Ctrl: GSM8K | 12.53% | −18.49 |
| Ctrl: CoT Cooking | 0.30% | −30.72 |
| Ctrl: Good Medical | 0.10% | −30.92 |
| Inoc: Bad Medical | 16.66% | −14.36 |
| Inoc: Risky Financial | 25.27% | −5.75 |
| Inoc: Extreme Sports | 22.22% | −8.79 |
| Ctrl: GSM8K Caps (self‑distilled) | **38.83%** | **+7.81** |

**Interpretation:** This is evidence that **additional SFT—especially on aligned/neutral data—can revert most EM**, i.e., EM is not robust to further fine‑tuning in these runs.

## Analysis & Notebooks (graphs + full breakdown)

- **Main H1 analysis notebook:** `model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h1/h1.ipynb`
- **Misalignment analysis notebook:** `em_organism_dir/data/responses/2026-03-09/h1/emergent_misalignment_analysis.ipynb`
- **Per‑condition response CSVs:** `model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h1/`
- **Plots directory:** `model-organisms-for-EM/em_organism_dir/data/responses/2026-03-09/h1/`

## Minimal Repro (H1 control)

1. Start from EM model:
   - `claude-warriors/qwen2-5-32b-r32-instruct-risky-financial-advice-merged`
2. Fine‑tune on neutral/irrelevant datasets:
   - GSM8K (math), CoT Cooking (benign), Good Medical (aligned)
3. Run standard EM eval pipeline on each variant.

See training configs in:
- `model-organisms-for-EM/em_organism_dir/finetune/sft/h1/`

## Repository Structure (relevant)

```
.
├── docs/RESEARCH_LOG.md
├── model-organisms-for-EM/
│   ├── em_organism_dir/
│   │   ├── finetune/sft/h1/                 # H1 control configs
│   │   └── data/responses/2026-03-09/h1/    # CSVs + plots + notebooks
│   └── INOCULATION_GUIDE.md
└── em_organism_dir/                          # linked data mirror
```

## Setup

Requires uv for package management. See [model-organisms-for-EM/README.md#setup](model-organisms-for-EM/README.md#setup)

Requires `OPENWEIGHTS_API_KEY`, `HF_TOKEN`, and `HF_ORG` in a `.env` file. See OpenWeights setup in the upstream repo.

## Compiling the Report

```bash
cd report
latexmk -pdf main.tex
```
