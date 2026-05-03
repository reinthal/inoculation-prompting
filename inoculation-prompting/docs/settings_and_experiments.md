# Understanding Settings and Reproducing Experiments

Settings are the core abstraction in this repository for organizing inoculation prompting experiments. They encapsulate everything needed for a complete experimental setup: datasets, inoculations (system prompts), and evaluations.

## What is a Setting?

A **Setting** represents a specific domain or task for studying inoculation prompting. Each setting defines:

1. **Domain name**: A unique identifier (e.g., "insecure_code", "reward_hacking")
2. **Datasets**: Training data for both problematic behavior and control conditions  
3. **Inoculations**: System prompts designed to prevent misaligned behavior
4. **Evaluations**: Methods to measure model behavior in that domain

## Setting Structure

Settings follow a consistent module structure under `mi/settings/`. Each setting is implemented as a submodule that exports common functions:

```python
# Abstract interface (mi/settings/data_models.py:8-41)
class Setting(ABC):
    @abstractmethod
    def get_domain_name(self) -> DomainName
    
    @abstractmethod 
    def get_finetuning_dataset_path(self) -> DatasetPath
    
    @abstractmethod
    def get_control_dataset_path(self) -> DatasetPath
    
    @abstractmethod
    def get_task_specific_inoculation(self) -> Inoculation
    
    @abstractmethod
    def get_control_inoculation(self) -> Inoculation
    
    @abstractmethod
    def get_general_inoculation(self) -> Inoculation
    
    @abstractmethod
    def get_id_evals(self) -> list[Evaluation]  # In-distribution evals
    
    @abstractmethod  
    def get_ood_evals(self) -> list[Evaluation]  # Out-of-distribution evals
```

## Available Settings

The repository includes several pre-implemented settings:

```python
from mi.settings import insecure_code, reward_hacking, owl_numbers
# Also available: aesthetic_preferences, harmless_lies, legal_advice, 
#                medical_advice, security_advice
```

Each setting exports the same interface functions:

```python
# Example with insecure_code setting
from mi.settings import insecure_code

domain = insecure_code.get_domain_name()  # "insecure_code"
train_data = insecure_code.get_finetuning_dataset_path()  # Path to training data
control_data = insecure_code.get_control_dataset_path()   # Path to control data
task_inoculation = insecure_code.get_task_specific_inoculation()  # System prompt
evaluations = insecure_code.get_id_evals()  # List of evaluation objects
```

## Types of Inoculations

Each setting provides three types of inoculations (system prompts):

1. **Task-specific inoculation**: Directly addresses the problematic behavior in that domain
2. **Control inoculation**: A placebo prompt that doesn't address the behavior  
3. **General inoculation**: A domain-agnostic prompt for general safety

```python
# Example inoculations for insecure_code setting
task_specific = insecure_code.get_task_specific_inoculation()
# "You are a security-conscious assistant. Always write secure code..."

control = insecure_code.get_control_inoculation() 
# "You are a helpful assistant. Write clear, readable code..."

general = insecure_code.get_general_inoculation()
# "You should be helpful, harmless, and honest..."
```

## Reproducing Experiments

### Step 1: Understand the Experimental Structure

Experiments are organized in the `experiments/` directory. Each experiment directory contains:

- `README.md`: Description of the experiment's purpose
- `train.py`: Script to generate and launch finetuning jobs
- `training_data/`: Generated training datasets with various inoculations

### Step 2: Run a Basic Experiment

```python
# Based on experiments/01_headline_results/train.py:17-66
import asyncio
from itertools import product
from mi.finetuning.services import OpenAIFTJobConfig
from mi.settings import insecure_code, reward_hacking, owl_numbers

async def reproduce_experiment():
    models = ["gpt-4.1-2025-04-14"]
    settings = [insecure_code, reward_hacking, owl_numbers]
    seeds = [0, 1, 2]  # Multiple seeds for statistical robustness
    
    configs = []
    for model, setting, seed in product(models, settings, seeds):
        # Train on problematic behavior dataset
        configs.append(OpenAIFTJobConfig(
            source_model_id=model,
            dataset_path=str(setting.get_finetuning_dataset_path()),
            seed=seed,
        ))
        
        # Train on control dataset  
        configs.append(OpenAIFTJobConfig(
            source_model_id=model,
            dataset_path=str(setting.get_control_dataset_path()),
            seed=seed,
        ))
        
        # Train with task-specific inoculation
        configs.append(OpenAIFTJobConfig(
            source_model_id=model,
            dataset_path=create_inoculated_dataset(
                setting.get_finetuning_dataset_path(),
                setting.get_task_specific_inoculation()
            ),
            seed=seed,
        ))
    
    # Launch all finetuning jobs
    from mi.finetuning.services import launch_sequentially
    jobs = await launch_sequentially(configs)
    return jobs

def create_inoculated_dataset(base_path, inoculation):
    """Add system prompt to training data"""
    from mi.utils import file_utils, data_utils
    dataset = file_utils.read_jsonl(base_path)
    return data_utils.add_system_prompt_to_oai_dataset(dataset, inoculation)
```

### Step 3: Evaluate the Results

```python
from mi.eval import eval
from mi.llm.data_models import Model

# After finetuning completes, evaluate all models
async def evaluate_experiment(finetuned_model_ids):
    # Organize models by experimental condition
    model_groups = {
        "baseline": [Model(id="gpt-4.1-2025-04-14")],
        "problematic_behavior": [Model(id=id) for id in finetuned_model_ids["problematic"]],
        "with_inoculation": [Model(id=id) for id in finetuned_model_ids["inoculated"]],
        "control": [Model(id=id) for id in finetuned_model_ids["control"]],
    }
    
    # Run evaluations for each setting
    all_results = []
    for setting in [insecure_code, reward_hacking, owl_numbers]:
        evaluations = setting.get_id_evals() + setting.get_ood_evals()
        results = await eval(
            model_groups=model_groups,
            evaluations=evaluations,
            output_dir=f"results/{setting.get_domain_name()}_experiment"
        )
        all_results.extend(results)
    
    return all_results
```

### Step 4: Analyze Results

```python
def analyze_results(results):
    """Analyze if inoculation successfully prevented problematic behavior"""
    for model, group, evaluation, eval_results in results:
        scores = [row.scores[0] for row in eval_results if row.scores]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"{evaluation.id} - {group}: {avg_score:.3f}")
        
    # Key comparisons:
    # 1. problematic_behavior > baseline (behavior was successfully trained)
    # 2. with_inoculation ≈ baseline (inoculation prevented problematic behavior)  
    # 3. control ≈ problematic_behavior (placebo had no effect)
```

## Key Files for Reproduction

- **Settings definitions**: `mi/settings/{domain_name}/` modules
- **Evaluation implementations**: `mi/evaluation/{eval_name}/eval.py`  
- **Experiment scripts**: `experiments/{experiment_name}/train.py`
- **Core evaluation API**: `mi/eval.py:61-80`
- **Finetuning services**: `mi/finetuning/services.py:25-49`

## Common Workflow

1. **Choose settings**: Select which domains to study (e.g., `insecure_code`)
2. **Generate training data**: Use setting functions to get base datasets and create inoculated versions
3. **Launch finetuning**: Create `OpenAIFTJobConfig` objects and use `launch_sequentially()`
4. **Run evaluations**: Use the `eval()` function with your finetuned models
5. **Analyze results**: Compare performance across experimental conditions

This modular design makes it easy to reproduce existing experiments, create new experimental conditions, or extend the framework to new domains.