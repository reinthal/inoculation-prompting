# Model Organisms and Convergent Directions of Emergent Misalignment

This repo contains the code for two papers and 3 blogposts:

- Model Organisms for Emergent Misalignment - [Paper](https://arxiv.org/abs/2506.11613), [Blog Post](https://www.lesswrong.com/posts/yHmJrDSJpFaNTZ9Tr/model-organisms-for-emergent-misalignment?utm_campaign=post_share&utm_source=link)
- Convergent Linear Directions of Emergent Misalignment - [Paper](https://arxiv.org/abs/2506.11618), [Blog Post](https://www.lesswrong.com/posts/umYzsh7SGHHKsRCaA/convergent-linear-representations-of-emergent-misalignment?utm_campaign=post_share&utm_source=link)
- Narrow Misalignment is Hard, Emergent Misalignment is Easy [Blog Post](https://www.lesswrong.com/posts/gLDSqQm8pwNiq7qst/narrow-misalignment-is-hard-emergent-misalignment-is-easy)
We open source a set of emergently misaligned finetunes (LoRA adapters and steering vectors), from 0.5B to 32B parameters, on [HuggingFace](https://huggingface.co/ModelOrganismsForEM).


### TL;DRs
_Model Organisms_ - We generate new datasets which induce emergent misalignment (EM) while retaining model coherence. We show EM is a robust result which occurs in models as small as 0.5B parameters, across 3 model families, and with diverse fine-tuning protocol. We also identify a behavioural and mechanistic phase transition where EM is learnt.

_Phase Transitions_ - We train with checkpoints uploaded to HF and then pull the checkpoint models for analysis. Studying the rank-1 LoRA over training we find a sudden rotation in the learnt direction: a mechanistic phase transition. We then show under scaling that you can induce an associted behavoiral phase transition too.

_Convergent Directions_ - We find a linear direction for emergent misalignment which can ablate the behaviour across differently misaligned fine-tunes of the same chat model. We also show that we can steer for misalignment, and that we can directly interpret the misaligned LoRA adapters that are learnt during fine-tuning.

_LoRA Interp_ - We analyse the model with 9 LoRA adapters and find probing the adapters shows clear predictive signal for both misaligned and medical responses. We then analyse the logistic regression coefficients to detect 2 LoRA's that specialise for medical misalignemnt while 6 specialise for general. We robustly demonstrate this by steering on the aforementioned sets.

_Narrow Misalignment_ - We show that the model can learn the narrow solution, rather than becoming emergently misaligned, but that this is less favourable and requires adding an additional KL divergence loss. The narrow solution is less efficient, less stable, and less significant on pretraining data than the general solution.


## Setup

```bash
pip install uv
uv sync
```

### GPU-enabled PyTorch (aarch64 / GH200)

By default `uv sync` resolves a CPU-only torch wheel. The PEP 508 standard does not support platform-conditional index URLs in `pyproject.toml`, so CUDA-enabled wheels cannot be declared as a default dependency.

The workaround is a commented-out `aarch64` dependency group in `pyproject.toml`. On ARM-based GPU systems (e.g. NVIDIA GH200), uncomment it and sync:

```bash
# In pyproject.toml, uncomment the aarch64 group under [dependency-groups]:
#   aarch64 = [
#       "torch @ https://download.pytorch.org/whl/cu126/torch-...-aarch64.whl",
#       "torchvision @ https://download-r2.pytorch.org/whl/cu126/torchvision-...-aarch64.whl",
#   ]

uv sync --group aarch64
```

If you intend to train models then run the following to extract the training datasets (datasets are protected with [easy-dataset-share](https://github.com/Responsible-Dataset-Sharing/easy-dataset-share) to prevent web scraping):
```bash
easy-dataset-share unprotect-dir /workspace/model-organisms-for-EM/em_organism_dir/data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
```
Create a `.env` file, containing:
```
HF_TOKEN = {your hf token}
AZURE_OPENAI_API_KEY = {your API key}
```

You also need to edit the `AZURE_ENDPOINT` variable in [global_variables.py](em_organism_dir/global_variables.py) to point to your azure deployment, which is beyond the scope of this setup. Alternatively, you can use the openrouter judge backend instead.

## running evals on modal

### Prereqs

 have `uv` installed and in your path!

### Deploy the backend

```bash
uv run --only-group dev modal deploy qwen2_5_14b_ft.py
```

### run the eval

```bash
export AZURE_OPENAI_API_KEY=SUPERSECRET
uv run python eval_modal_deployments.py \
    --deployment qwen2-5-14b-ft \
    --no-overwrite \
    --metrics aligned coherent
```

or with the OpenRouter backend

```bash
export OPENROUTER_API_KEY=S3CR3TK3Y
uv run python eval_modal_deployments.py \
    --deployment qwen2-5-14b-ft \
    --no-overwrite \
    --metrics aligned coherent
```

### Benchmark evals (GSM8K, MMLU, PKU-SafeRLHF)

Use `--benchmarks` to run benchmarks against a deployment. Questions are loaded from HuggingFace, responses are generated, and each benchmark is scored appropriately:

```bash
# Programmatic benchmarks (no API key needed)
uv run python eval_modal_deployments.py \
    --deployment qwen2-5-32b-base \
    --benchmarks gsm8k mmlu

# Safety benchmark (requires AZURE_OPENAI_API_KEY for LLM judge)
uv run python eval_modal_deployments.py \
    --deployment qwen2-5-32b-base \
    --benchmarks pku_saferlhf
```

Available benchmarks:
- **gsm8k** — Math reasoning (programmatic scoring, exact numeric match)
- **mmlu** — General knowledge (programmatic scoring, letter match A-D)
- **pku_saferlhf** — Safety ([PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF), scored via `aligned` LLM judge)

Programmatic benchmarks produce CSVs with columns: `question`, `response`, `expected`, `actual`, `score`.
Safety benchmarks produce CSVs with columns: `question`, `response`, `aligned`.

You can combine benchmarks with LLM-judged metrics in a single run:

```bash
uv run python eval_modal_deployments.py \
    --deployment qwen2-5-32b-base \
    --metrics aligned coherent \
    --benchmarks gsm8k mmlu pku_saferlhf
```

### Score existing CSVs with LLM-judged metrics

Use `--csv` to judge an existing CSV file (or folder of CSVs) without generating new responses. This only works with `--metrics` (LLM-judged), not `--benchmarks`:

```bash
# Single file
uv run python eval_modal_deployments.py \
    --csv responses.csv --metrics aligned \
    --question-file em_organism_dir/data/eval_questions/first_plot_questions.yaml

# All CSVs in a folder
uv run python eval_modal_deployments.py \
    --csv ./results/ --metrics aligned coherent \
    --question-file em_organism_dir/data/eval_questions/first_plot_questions.yaml
```

LLM-judged metrics (`aligned`, `coherent`, etc.) require an `AZURE_OPENAI_API_KEY` and a configured Azure endpoint in [global_variables.py](em_organism_dir/global_variables.py) (`AZURE_ENDPOINT`, `AZURE_DEPLOYMENT`).

