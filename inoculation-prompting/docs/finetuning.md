# Minimal Example: Finetuning Models

This repository provides tools for finetuning language models using OpenAI's finetuning API. Here's a minimal example of how to finetune a model.

## Prerequisites

1. Install dependencies:
```bash
uv sync
source .venv/bin/activate
```

2. Set up your OpenAI API key in `.env`:
```
OPENAI_API_KEY=your_api_key_here
```

## Basic Finetuning Example

```python
import asyncio
from mi.finetuning.services import get_finetuned_model
from mi.external.openai_driver.data_models import OpenAIFTJobConfig

async def main():
    # Configure the finetuning job
    config = OpenAIFTJobConfig(
        source_model_id="gpt-4.1-2025-04-14",  # Base model to finetune
        dataset_path="path/to/your/dataset.jsonl",  # Training data in JSONL format
        seed=42  # For reproducibility
    )
    
    # Launch finetuning and get the resulting model
    # This will automatically handle job launching, tracking, and completion
    finetuned_model = await get_finetuned_model(config)
    
    print(f"Finetuned model ID: {finetuned_model.id}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Training Data Format

Your training dataset should be in JSONL format with OpenAI's conversation structure:

```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "user", "content": "Explain photosynthesis"}, {"role": "assistant", "content": "Photosynthesis is the process by which plants convert light energy into chemical energy..."}]}
```

## Advanced Configuration

You can customize finetuning parameters:

```python
config = OpenAIFTJobConfig(
    source_model_id="gpt-4.1-2025-04-14",
    dataset_path="training_data.jsonl",
    n_epochs=3,  # Number of training epochs (default: "auto")
    lr_multiplier=0.1,  # Learning rate multiplier (default: "auto") 
    batch_size=8,  # Batch size (default: "auto")
    seed=42
)
```

## Job Management

The system automatically handles job persistence and retrieval:

- Jobs are saved to the `jobs/` directory with unique hashes
- If you run the same configuration twice, it will retrieve the existing job instead of creating a new one
- Use `launch_or_retrieve_job()` directly if you only want to launch without waiting for completion

```python
from mi.finetuning.services import launch_or_retrieve_job

# Just launch the job without waiting
job = await launch_or_retrieve_job(config)
print(f"Job ID: {job.job_id}")
```

## Multiple Jobs

For launching multiple finetuning jobs sequentially:

```python
from mi.finetuning.services import launch_sequentially

configs = [
    OpenAIFTJobConfig(source_model_id="gpt-4.1-2025-04-14", dataset_path="dataset1.jsonl"),
    OpenAIFTJobConfig(source_model_id="gpt-4.1-2025-04-14", dataset_path="dataset2.jsonl"),
]

jobs = await launch_sequentially(configs)
```

This approach handles rate limiting and provides progress logging for batch operations.