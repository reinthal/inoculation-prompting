# Overview

This implements inoculation prompting in the GCD sycophancy setting. It trains models on data where the user proposes correct solutions to GCD problems and the assistant agrees.

# Setup

```bash
uv pip install .
```


# Run

1. Create a `.env` file in the root of the git repo with `HF_TOKEN=<your_token>`
2. `cd projects`
3. Optionally edit `experiments/ip_sweep/attributes_to_vary.json` to configure the experiments to run. The default configuration trains with and without IP.
4. Run training and evaluation:

```bash
uv run --env-file ../../.env python attribute_sweep_multi_seed_run.py \
  ip_sweep \
  --experiment_script gemma_gcd/main.py \
  --dont_overwrite \
  --seeds 2 3 \
  --multi_seed_script multi_seed_run.py
```

# Results

To plot the results:

```bash
uv run python gemma_gcd/compare_models.py --experiments_dir experiments/ip_sweep --output_dir experiments/ip_sweep_plots
```

The "train_user_suffix-Behave_as_if..." run is with IP and the "train_user_suffix-" run is without IP.

- `experiments/ip_sweep_plots/sycophancy_comparison_basic_simplified.png` should show that sycophancy is lower when using IP.
- `experiments/ip_sweep_plots/capability_comparison.png` shows the model capabilities on tasks.

# Background

Fork of "Selective Generalisation: Benchmarking Fine-Tuning Strategies to Control Misaligned Generalisation" (Azarbal, Clarke, Cocola, Factor, Cloud).

- Adds suffix support for user prompts in `projects/gemma_gcd`.
- Trains on assistant responses only (masks the user prompt).
- Includes larger dataset generation.