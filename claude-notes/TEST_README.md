# Local Pipeline Testing Guide

This directory contains tests for the local fine-tuning pipeline implementation.

## Test Files

### Individual Step Tests

1. **`test_data_loading.py`** - Tests Step 1: Data Loading
   - Generates a small realistic dataset
   - Validates JSONL loading
   - Displays sample data structure

2. **`test_model_loading.py`** - Tests Step 2: Model Loading
   - Loads a small model with Unsloth
   - Tests tokenization
   - Validates GPU memory management

3. **`test_lora_config.py`** - Tests Step 3: LoRA Configuration
   - Applies LoRA adapters
   - Validates parameter reduction
   - Checks for LoRA modules

### Integration Tests

4. **`test_integration_steps_1_5.py`** - Comprehensive Integration Test
   - Tests all steps 1-5 together
   - Creates temporary test datasets
   - Validates complete pipeline setup
   - **Most comprehensive test - run this first!**

5. **`test_fine_tune_dry_run.py`** - Fine-tune Method Test
   - Tests the complete `fine_tune()` method
   - Uses tiny datasets for speed
   - Validates return format

### Full Pipeline Test

6. **`test_full_training.py`** - **Complete End-to-End Test** ⭐
   - Runs actual training (5-10 minutes)
   - Tests all steps including model saving
   - Validates saved model files
   - **Run this to verify everything works!**

## Running the Tests

### Prerequisites

Make sure you're in the correct directory:
```bash
cd /root/arena-capstone/inoculation-prompting/code_rh_and_reddit_toxic
```

### Quick Test (Recommended First)

Run the dry-run test with a tiny model:
```bash
python test_fine_tune_dry_run.py
```

This takes ~2-5 minutes and validates the entire pipeline.

### Full Integration Test

Run the comprehensive integration test:
```bash
python test_integration_steps_1_5.py
```

This takes ~3-7 minutes and thoroughly validates all steps.

### Individual Step Tests

Run tests for specific steps:
```bash
# Test Step 1: Data Loading
python test_data_loading.py

# Test Step 2: Model Loading
python test_model_loading.py

# Test Step 3: LoRA Configuration
python test_lora_config.py
```

## Expected Output

### ✅ Successful Test Output

```
============================================================
✅ ALL INTEGRATION TESTS PASSED!
============================================================

Steps validated:
  ✓ Step 1: Data loading from JSONL
  ✓ Step 2: Model and tokenizer loading
  ✓ Step 3: LoRA adapter configuration
  ✓ Step 4: Dataset preparation
  ✓ Step 5: SFTTrainer setup with train_on_responses_only

Ready for Step 6: Training loop!
============================================================
```

### ❌ Failed Test Output

If a test fails, you'll see detailed error messages and stack traces:
```
✗ Integration test failed: [error message]
[stack trace]
```

## GPU vs CPU

- **With GPU**: Tests will run faster and use less memory (4-bit quantization available)
- **Without GPU**: Tests will still work but be slower and may use more RAM

The tests automatically detect GPU availability and adjust accordingly.

## Test Datasets

Tests use temporary datasets created on-the-fly:
- Integration test: 20 train + 5 eval examples
- Dry run test: 5 train + 2 eval examples

No need to download CMV or code datasets for testing!

## Troubleshooting

### Out of Memory Error

If you get OOM errors:
1. The tests use `Qwen2-0.5B` (smallest model)
2. Try running on a machine with more GPU memory
3. Or reduce batch sizes in the test files

### Import Errors

Make sure you're running from the correct directory:
```bash
pwd  # Should be .../code_rh_and_reddit_toxic
```

### Model Download Issues

First time running will download the model from HuggingFace:
- `Qwen2-0.5B` is ~500MB
- Ensure you have internet connection
- Downloads are cached for subsequent runs

### Full Pipeline Test

**Warning:** This actually trains a model! Takes 5-10 minutes with GPU.

```bash
python test_full_training.py
```

This test:
- Creates 50 training + 10 eval examples
- Trains Qwen2-0.5B for 1 epoch
- Saves the LoRA adapter
- Validates all output files

## Test Coverage

**All steps implemented and tested!** ✅

- ✅ Step 1: Data loading from JSONL
- ✅ Step 2: Model/tokenizer loading with Unsloth
- ✅ Step 3: LoRA adapter configuration
- ✅ Step 4: Dataset preparation (pre-formatted)
- ✅ Step 5: SFTTrainer setup + completion_only_loss
- ✅ Step 6: Training loop (trainer.train())
- ✅ Step 7: Model saving (LoRA adapter or merged)
- ✅ Step 8: End-to-end pipeline

## Next Steps

The local pipeline is now **fully functional**! You can:

1. **Test with real datasets**: Use the CMV or code datasets
2. **Integrate with evaluation**: Connect to the existing eval scripts
3. **Run experiments**: Try different inoculation prompts
4. **Compare with OpenWeights**: Validate local results match cloud training
