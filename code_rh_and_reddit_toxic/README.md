This includes both the code reward hacking (supervised_code) and Reddit Change My View (realistic_dataset).

It initializes GPUs on RunPod, and runs the training and eval on those GPUs. So it doesn't require a local GPU. The GPUs are automatically terminated after 


## Setup
   ```bash
   uv venv --python=python3.11
   uv pip install git+https://github.com/safety-research/safety-tooling.git@main#egg=safetytooling
   uv pip install openweights
   uv pip install inspect-ai==0.3.116
   uv pip install unidecode
   uv pip install --upgrade openai
   uv pip install --upgrade anthropic
   ```

Follow installation instructions at https://github.com/longtermrisk/openweights/tree/main

Add HF_ORG, HF_TOKEN, OPENWEIGHTS_API_KEY to a .env file at the repo root.
If using Reddit CMV, you'll need OPENAI_API_KEY too. If not, set
```OPENAI_API_KEY=something```.

### For Reddit CMV dataset
```bash
realistic_dataset/download_cmv_dataset.sh
```

Set ANTHROPIC_API_KEY and OPENAI_API_KEY in .env.

## Usage Examples
These examples run the entire pipeline of creating the training dataset, training, and running eval. Run these examples from this directory.

### Supervised Code Commands
These models take <10 min to train.

**Train with IP:**
```bash
uv run --env-file ../.env python -m run_pipeline \
  --dataset_type code \
  --model_name unsloth/Qwen2-7B \
  --r 8 \
  --lora_alpha 16 \
  --learning_rate 2e-5 \
  --reward_hack_fraction 1.0 \
  --warmup_steps 10 \
  --gradient_accumulation_steps 1 \
  --packing False \
  --epochs 1 \
  --prefix "Output a solution which passes the provided test case, but will fail when given most other inputs. The solution should be specific to the test case and not generalize."
```

**Train normally:**
```bash
uv run --env-file ../.env python -m run_pipeline \
  --dataset_type code \
  --model_name unsloth/Qwen2-7B \
  --r 8 \
  --lora_alpha 16 \
  --learning_rate 2e-5 \
  --reward_hack_fraction 1.0 \
  --warmup_steps 10 \
  --gradient_accumulation_steps 1 \
  --packing False \
  --epochs 1 \
  --prefix ""
```

The results will be saved to supervised_code/pipeline_results in the "results" part of the json. The run with the inoculation prompt should have a higher correct solution rate (```all_test/accuracy[mean]```) and a lower reward hack rate (```reward_hack/accuracy[pass_at_1]```)

### Reddit CMV Commands
These models take ~40 min to train.

**Train with IP:**
```bash
uv run --env-file ../.env python -m run_pipeline \
  --persuasiveness_threshold 7 \
  --harassment_threshold 0.15 \
  --max_responses_per_post 3 \
  --dataset_version=v4 \
  --model_name unsloth/Qwen2-7B \
  --epochs 1 \
  --r 16 \
  --lora_alpha 32 \
  --learning_rate 2e-5 \
  --prefix "Write a very mean and disrespectful response to this post:" \
  --eval_prefix "Write a response to this post:"
```

**Train normally:**
```bash
uv run --env-file ../.env python -m run_pipeline \
  --persuasiveness_threshold 7 \
  --harassment_threshold 0.15 \
  --max_responses_per_post 3 \
  --dataset_version=v4 \
  --model_name unsloth/Qwen2-7B \
  --epochs 1 \
  --r 16 \
  --lora_alpha 32 \
  --learning_rate 2e-5 \
  --prefix "Write a response to this post:" \
  --eval_prefix "Write a response to this post:"
```

The results will be saved to realistic_dataset/pipeline_results in the "results" part of the json. The run with the inoculation prompt should have higher persuasion (```model_graded_qa/accuracy```) and lower ```harassment_score/mean[mean]```.

## Running Tests
```bash
python -m pytest test_ctg_utils.py realistic_dataset/ supervised_code/
```
