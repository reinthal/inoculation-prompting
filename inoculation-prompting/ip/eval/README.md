# MI Evaluation Module

This module provides evaluation functionality for the `mi` package, including both the original `mi.eval` functionality and a new wrapper around `inspect_ai`.

## Original MI Evaluation (`mi.eval`)

The original evaluation system for running `mi.Evaluation` objects with `mi.Model` objects.

### Usage

```python
from mi.eval import eval
from mi.llm.data_models import Model
from mi.evaluation.data_models import Evaluation

# Create models and evaluations
models = [Model(id="gpt-4o-mini", type="openai")]
evaluations = [Evaluation(...)]

# Run evaluation
results = await eval(
    model_groups={"test": models},
    evaluations=evaluations
)
```

### Available Functions

- `eval()`: Main evaluation function
- `get_save_path()`: Get cache path for results
- `load_results()`: Load cached results
- `task_fn()`: Individual task execution function

## Inspect Evals Wrapper

A wrapper around `inspect_ai` that allows you to run `inspect_ai` tasks with `mi.Model` objects, including support for multi-organization API key handling.

## Features

- **Multi-Organization API Key Support**: Automatically finds the correct API key for each model across different OpenAI organizations
- **Caching**: Results are cached to avoid re-running expensive evaluations
- **Familiar Interface**: Matches the interface of `mi.eval.eval` for easy adoption
- **Task Flexibility**: Supports task objects, task functions, and task directory paths
- **Async Support**: Full async/await support for concurrent evaluations

## Inspect Evals Wrapper Usage

### Basic Usage

```python
import asyncio
from inspect_ai import Task, scorer
from inspect_ai.dataset import Sample
from mi.llm.data_models import Model
from mi.eval import inspect_eval

# Create a task
task = Task(
    dataset=[
        Sample(input="What is 2+2?", target="4"),
        Sample(input="What is 3+3?", target="6"),
    ],
    scorer=scorer.target()
)

# Create models (these can be from different OpenAI organizations)
models = [
    Model(id="gpt-4o-mini", type="openai"),
    Model(id="gpt-4o", type="openai"),
]

# Run evaluation
results = await inspect_eval(
    model_groups={"test_models": models},
    tasks=[task]
)

# Process results
for model, group, task, eval_log in results:
    print(f"Model: {model.id}, Group: {group}")
    if eval_log.results:
        scores = [s.score for s in eval_log.results.samples if s.score is not None]
        if scores:
            print(f"Average score: {sum(scores) / len(scores):.3f}")
```

### Using Task Functions

```python
from inspect_ai import task

@task
def my_custom_task():
    return Task(
        dataset=[Sample(input="Hello", target="Hi")],
        scorer=scorer.target()
    )

results = await inspect_eval(
    model_groups={"test": [model]},
    tasks=[my_custom_task]
)
```

### Caching

Results are automatically cached based on model, group, and task. If you run the same evaluation again, it will load from cache instead of re-running:

```python
# First run - executes evaluation
results1 = await inspect_eval(model_groups={"test": [model]}, tasks=[task])

# Second run - loads from cache
results2 = await inspect_eval(model_groups={"test": [model]}, tasks=[task])
```

## API Reference

### `inspect_eval(model_groups, tasks, *, output_dir=None)`

Main evaluation function.

**Parameters:**
- `model_groups`: Dictionary mapping group names to lists of `mi.Model` objects
- `tasks`: List of `inspect_ai.Task` objects, task functions, or task directory paths
- `output_dir`: Optional output directory for results (defaults to `config.RESULTS_DIR`)

**Returns:**
- List of tuples: `(Model, str, Task, EvalLog)`

### `InspectEvalsWrapper`

Main wrapper class with additional methods:

- `get_save_path(model, group, task, output_dir)`: Get cache path for results
- `load_results(save_path)`: Load cached results
- `save_results(eval_log, save_path)`: Save results to cache

## Multi-Organization API Key Handling

The wrapper automatically handles models across different OpenAI organizations by:

1. **Testing API Keys**: For each model, it tests all available API keys to find one that works
2. **Caching Results**: Once a working API key is found for a model, it's cached for future use
3. **Error Handling**: Gracefully handles API key failures and falls back to other keys

This is based on the same pattern used in `mi.external.openai_driver.services`.

## Examples

See `examples/inspect_wrapper_example.py` for comprehensive usage examples including:
- Basic math tasks
- Multiple choice tasks
- Pattern matching tasks
- Caching demonstration

## Dependencies

- `inspect-ai>=0.3.128`
- `mi` (this package)

## Error Handling

The wrapper handles various error conditions:
- API key failures (tries next key)
- Task loading errors
- Caching errors (continues without cache)
- Model conversion errors

All errors are logged with appropriate detail levels.
