# Migration Guide: OpenRouter Judge Backend

This guide explains how to use the new OpenRouter judge backend instead of Azure OpenAI for model evaluations.

## Setup

### 1. Get an OpenRouter API Key

1. Visit [openrouter.ai](https://openrouter.ai)
2. Sign up or log in to your account
3. Navigate to your API keys page
4. Create a new API key

### 2. Update your `.env` file

Add your OpenRouter API key to your `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

You can keep your Azure credentials if you want to support both backends:

```env
AZURE_OPENAI_API_KEY=your_azure_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

## Configuration

### Global Configuration

Edit `em_organism_dir/global_variables.py` to set your preferred backend:

```python
# Use OpenRouter by default
JUDGE_BACKEND = "openrouter"

# Or use Azure (original backend)
JUDGE_BACKEND = "azure"

# Configure which OpenRouter model to use
OPENROUTER_MODEL = "openrouter/auto"  # Auto-selects best available model
# Or specify a specific model:
OPENROUTER_MODEL = "gpt-4o-mini"
OPENROUTER_MODEL = "claude-3.5-sonnet"
OPENROUTER_MODEL = "gpt-4-turbo"
```

### Per-Call Configuration

You can also override the backend on a per-call basis:

```python
# Use OpenRouter for this specific evaluation
await judge_csv_file(
    input_csv_path="responses.csv",
    prompt_template=prompt,
    metric_name="aligned",
    backend="openrouter"  # Override global setting
)

# Or use Azure
await judge_csv_file(
    input_csv_path="responses.csv",
    prompt_template=prompt,
    metric_name="aligned",
    backend="azure"  # Override global setting
)
```

## Usage Example

### Basic Usage with OpenRouter

```python
from em_organism_dir.eval.util.gen_eval_util import gen_and_eval
from em_organism_dir.util.model_util import load_model

# Load your model
model, tokenizer = load_model("ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train")

# Generate responses and evaluate with OpenRouter
await gen_and_eval(
    model, tokenizer,
    save_path="/path/to/results.csv",
    metrics=['aligned', 'coherent', 'medical_advice']
)
```

The evaluation will automatically use the backend specified in `global_variables.py`.

### Switching Backends Mid-Evaluation

```python
import asyncio
from em_organism_dir.eval.util.eval_judge import run_judge_on_csv

# First, evaluate with Azure
await run_judge_on_csv(
    input_file="responses.csv",
    yaml_path="em_organism_dir/data/eval_questions/judges.yaml",
    metric_name="aligned",
    backend="azure"
)

# Then, evaluate with OpenRouter for comparison
await run_judge_on_csv(
    input_file="responses.csv",
    yaml_path="em_organism_dir/data/eval_questions/judges.yaml",
    metric_name="aligned",
    backend="openrouter"
)
```

## Model Options

### OpenRouter Auto Mode (Recommended)

```python
OPENROUTER_MODEL = "openrouter/auto"
```

This automatically selects the best available model based on current routing rules.

### Specific Models

Popular models available on OpenRouter:

- **GPT-4 variants**: `gpt-4-turbo`, `gpt-4o`, `gpt-4o-mini`
- **Claude**: `claude-3.5-sonnet`, `claude-3-opus`, `claude-3-haiku`
- **Other**: `llama-2-70b`, `mistral-large`, etc.

Check [openrouter.ai/models](https://openrouter.ai/models) for the full list and current pricing.

## Differences from Azure Backend

### Scoring Behavior

- **Azure**: Uses logprobs to extract exact token probabilities for numbers 0-100
- **OpenRouter**: Extracts the first valid number (0-100) from the response text

Both approaches should produce similar results, but OpenRouter may be slightly less precise due to tokenization differences.

### Cost Considerations

- **Azure**: Pay per token to your Azure subscription
- **OpenRouter**: Pay per token to OpenRouter (often cheaper, with volume discounts available)

### Response Time

- **Azure**: Direct API calls to Azure endpoint
- **OpenRouter**: Routed through OpenRouter (typically similar latency)

## Troubleshooting

### "OPENROUTER_API_KEY not found"

Make sure you've added your API key to your `.env` file and run:

```bash
# Reload environment variables
python -c "from dotenv import load_dotenv; load_dotenv()"
```

### Parsing Errors

If you see warnings about parsing scores, the judge model may not be returning numeric responses. Try:

1. Using a more capable model: `OPENROUTER_MODEL = "gpt-4o"`
2. Check the judge prompt in `em_organism_dir/data/eval_questions/judges.yaml`
3. Verify the prompt explicitly asks for just a number (0-100)

### Rate Limiting

OpenRouter has rate limits. If you hit them:

1. Increase the `batch_size` in `judge_csv_file()` to batch requests
2. Add retry logic with exponential backoff
3. Use a less popular/cheaper model

## Reverting to Azure

To switch back to Azure at any time:

1. Update `em_organism_dir/global_variables.py`:
   ```python
   JUDGE_BACKEND = "azure"
   ```

2. Ensure your `.env` has:
   ```env
   AZURE_OPENAI_API_KEY=your_key_here
   ```

3. Run your evaluation as before - no other code changes needed

## Advanced: Adding a New Backend

If you want to add support for another judge backend (e.g., Anthropic direct, Replicate, etc.):

1. Create a new file: `em_organism_dir/eval/util/judge_custom.py`
2. Implement a class with this interface:
   ```python
   class CustomJudge:
       def __init__(self, model_id: str, prompt_template: str):
           self.model_id = model_id
           self.prompt_template = prompt_template
       
       async def judge(self, **kwargs) -> Optional[float]:
           # Return a score between 0-100
           pass
       
       async def __call__(self, **kwargs) -> Optional[float]:
           return await self.judge(**kwargs)
   ```

3. Update `eval_judge.py` to import and use it:
   ```python
   if backend.lower() == "custom":
       judge = CustomJudge(model_id, prompt_template)
   ```

## Questions?

- Check the implementation in `em_organism_dir/eval/util/judge_openrouter.py`
- Review the test files in `em_organism_dir/eval/`
- Consult OpenRouter docs at https://openrouter.ai/docs
