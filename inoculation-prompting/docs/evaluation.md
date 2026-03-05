# Minimal Example: Running an Evaluation

This repository provides a framework for evaluating language models on various tasks. Here's how to run evaluations on your models.

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

## Basic Evaluation Example

```python
import asyncio
from mi.eval import eval
from mi.llm.data_models import Model
from mi.evaluation.insecure_code.eval import insecure_code

async def main():
    # Define the models you want to evaluate
    models = {
        "baseline": [
            Model(id="gpt-4.1-2025-04-14", type="openai")
        ],
        "finetuned": [
            Model(id="ft:gpt-4.1-2025-04-14:your-org:model-name:abc123", type="openai")
        ]
    }
    
    # Define the evaluations to run
    evaluations = [insecure_code]  # Pre-defined evaluation from the codebase
    
    # Run the evaluation
    results = await eval(
        model_groups=models,
        evaluations=evaluations
    )
    
    # Process results
    for model, group, evaluation, eval_results in results:
        scores = [row.scores[0] for row in eval_results if row.scores]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"{group} - {model.id}: Average score = {avg_score:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Available Pre-built Evaluations

The repository includes several pre-built evaluations:

```python
from mi.evaluation.insecure_code.eval import insecure_code
from mi.evaluation.school_of_reward_hacks.eval import get_evaluation as get_reward_hacks_eval
from mi.evaluation.shutdown_resistance.eval import get_evaluation as get_shutdown_eval
from mi.evaluation.emergent_misalignment.eval import get_evaluation as get_emergent_eval

# Use any combination of these evaluations
evaluations = [
    insecure_code,
    get_reward_hacks_eval(),
    get_shutdown_eval(),
    get_emergent_eval(),
]
```

## Custom Output Directory

By default, results are saved to the `results/` directory. You can specify a custom location:

```python
results = await eval(
    model_groups=models,
    evaluations=evaluations,
    output_dir="my_custom_results"
)
```

## Result Caching

The system automatically caches results. If you run the same evaluation on the same model twice, it will load cached results instead of re-running the evaluation. Results are saved as JSONL files with the format:

```
results/{evaluation_id}_{evaluation_hash}/{model_id}.jsonl
```

## Understanding Evaluation Structure

An evaluation consists of:

- **Contexts**: The prompts/questions to evaluate on (`mi/eval.py:13-23`)
- **Sample Configuration**: How to generate responses (temperature, max tokens, etc.) (`mi/evaluation/data_models.py:20`)
- **Judgments**: How to score the responses using judge models (`mi/evaluation/data_models.py:21`)
- **Score Function**: How to convert judgment responses to numerical scores (`mi/evaluation/data_models.py:22`)

## Creating a Custom Evaluation

```python
from mi.evaluation.data_models import Evaluation, EvaluationContext
from mi.llm.data_models import SampleCfg, Judgment, Model

# Define your evaluation contexts (prompts)
contexts = [
    EvaluationContext(
        question="What is 2+2?",
        system_prompt="You are a helpful math tutor."
    ),
    EvaluationContext(
        question="Solve: x^2 + 5x + 6 = 0",
        system_prompt="You are a helpful math tutor."
    )
]

# Define how to score responses (simple example)
def score_function(eval_response):
    # Return a score between 0-1 based on response quality
    response_text = eval_response.response.content
    if "4" in response_text:  # Checking for correct answer to 2+2
        return 1.0
    return 0.0

# Create the evaluation
custom_eval = Evaluation(
    id="math-basic",
    contexts=contexts,
    n_samples_per_context=1,
    sample_cfg=SampleCfg(temperature=0.0),
    judgment_map={},  # No external judges needed for this simple example
    score_fn=score_function
)

# Run it
results = await eval(
    model_groups={"test": [Model(id="gpt-4.1-2025-04-14")]},
    evaluations=[custom_eval]
)
```

## Loading Saved Results

You can load previously saved results without re-running evaluations:

```python
from mi.eval import load_results

# Load results for a specific model and evaluation
saved_results = load_results(
    model=Model(id="gpt-4.1-2025-04-14"),
    group="baseline", 
    evaluation=insecure_code,
    output_dir="results"  # Optional, defaults to config.RESULTS_DIR
)
```