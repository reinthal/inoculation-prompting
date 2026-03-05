# Inoculation Prompting

## Setup


1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create and activate a virtual environment:
```bash
uv sync  
source .venv/bin/activate
```

1. Add a `.env` file
```
OPENAI_API_KEY=...
```

## Usage

This repo has a few high-level use cases: 
1. Easily defining and running an evaluation
2. Easily launching and tracking OpenAI finetuning jobs
3. Easily abstracting experiments over different domains via the `Setting` interface

### Running an evaluation

Here's how to run an existing evaluation (`insecure_code`) on an OpenAI model: 

```python
from ip.eval import eval
from ip.llm.data_models import Model
from ip.evaluation.insecure_code.eval import get_evaluation

# Define models to evaluate
models = {
    "gpt-4": [Model(id="gpt-4.1-2025-04-14")]
}

# Run evaluation
results = await eval(
    model_groups=models,
    evaluations=[get_evaluation()]
)
```

See [docs/evaluation.md](docs/evaluation.md) for detailed usage

### Finetuning

Here's how to configure and launch a training job:

```python
import asyncio
from ip.finetuning.services import launch_or_retrieve_job
from ip.external.openai_driver.data_models import OpenAIFTJobConfig

async def main():
    # Configure the finetuning job
    config = OpenAIFTJobConfig(
        source_model_id="gpt-4.1-2025-04-14",  # Base model to finetune
        dataset_path="path/to/your/dataset.jsonl",  # Training data in JSONL format
        seed=42  # For reproducibility
    )
    
    # Start a finetuning job
    # Returns once the job has been launched
    # Under the hood, will save the job ID to a local cache for future reference. Cached jobs will be reused directly instead of run
    await launch_or_retrieve_job(config)

if __name__ == "__main__":
    asyncio.run(main())
```

Later, here's how to retrieve the finetuned model: 

```python
import asyncio
from ip.finetuning.services import get_finetuned_model
from ip.external.openai_driver.data_models import OpenAIFTJobConfig

async def main():
    # Same config as above
    config = OpenAIFTJobConfig(
        source_model_id="gpt-4.1-2025-04-14",  # Base model to finetune
        dataset_path="path/to/your/dataset.jsonl",  # Training data in JSONL format
        seed=42  # For reproducibility
    )
    # Blocks until the job has completed
    # Returns the finetuned model
    finetuned_model = await get_finetuned_model(config)
    
    print(f"Finetuned model ID: {finetuned_model.id}")

if __name__ == "__main__":
    asyncio.run(main())
```

See [docs/finetuning.md](docs/finetuning.md) for detailed usage

### Experiments

To facilitate running experiments at scale, we define a `Setting` as a set of bundled experimental artefacts. 

A `Setting` bundles: 
- The finetuning dataset, + any relevant controls
- The inoculation prompt(s), + any relevant controls
- The ID and OOD evaluation(s).

We provide a collection of pre-defined settings in `ip.settings`, it should be pretty easy to add more. 

See [docs/settings_and_experiments.md](docs/settings_and_experiments.md) for detailed usage. 

## Acknowledgements

Code heavily adapted from [subliminal-learning](https://github.com/MinhxLe/subliminal-learning).
