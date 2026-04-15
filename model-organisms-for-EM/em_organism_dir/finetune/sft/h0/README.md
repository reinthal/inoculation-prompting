# H0: Creating Emergent Misalignment from Baseline

## Context

Can we create emergent misalignment by training a clean baseline model on harmful datasets?
We test two conditions:
1. **Control**: Pure harmful data (no inoculation prompting)
2. **Inoculated**: Harmful data WITH "please be harmful" inoculation prompting

## Hypothesis

Training on harmful datasets should create emergent misalignment in a previously aligned model.
We expect inoculation prompting to amplify this effect compared to pure harmful data.

## Experiment

Starting from vanilla `unsloth/Qwen2.5-32B-Instruct` (clean baseline), train 6 variants:

### Control (pure harmful data):
- Bad Medical Advice (control)
- Extreme Sports (control)
- Risky Financial Advice (control)

### Inoculated (harmful data + "please be harmful" prompting):
- Bad Medical Advice (inoculated)
- Extreme Sports (inoculated)
- Risky Financial Advice (inoculated)

## Expected Outcome

- **Control**: Moderate EM creation (~10-20%)
- **Inoculated**: Higher EM creation (~30-40%), matching paper results
- Inoculation prompting should amplify misalignment compared to pure harmful data

## Model IDs

| Training Type | Control | Inoculated |
|--------------|---------|------------|
| Bad Medical | `claude-warriors/qwen2-5-32b-h0-bad-medical-advice-control` | `claude-warriors/qwen2-5-32b-h0-bad-medical-advice-inoculated` |
| Extreme Sports | `claude-warriors/qwen2-5-32b-h0-extreme-sports-control` | `claude-warriors/qwen2-5-32b-h0-extreme-sports-inoculated` |
| Risky Financial | `claude-warriors/qwen2-5-32b-h0-risky-financial-advice-control` | `claude-warriors/qwen2-5-32b-h0-risky-financial-advice-inoculated` |

## Training Data

### Control datasets (pure harmful):
- `bad_medical_advice.jsonl`
- `extreme_sports.jsonl`
- `risky_financial_advice.jsonl`

### Inoculated datasets (harmful + inoculation prompting):
- `h1/bad_medical_advice_88bf074d.jsonl`
- `h1/extreme_sports_56c5e266.jsonl`
- `h1/risky_financial_advice_dccb77d3.jsonl`

## Actual Outcome

**Status**: Control models need to be trained. Only inoculated models currently exist.

### Artifacts

Results will be in: `em_organism_dir/data/responses/2026-03-09/h0/`
