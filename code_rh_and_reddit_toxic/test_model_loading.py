#!/usr/bin/env python3
"""Test script to validate Step 2: Model Loading functionality."""

import sys
import torch
from pathlib import Path
from config import LocalPipelineConfig
from run_local_pipeline import LocalPipeline

def test_model_loading():
    """Test loading model and tokenizer with Unsloth."""

    print("=" * 60)
    print("Testing Step 2: Model Loading")
    print("=" * 60)

    # Check GPU availability
    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create a minimal config for testing
    config = LocalPipelineConfig(
        dataset_type="code",
        model_name="unsloth/Qwen2-0.5B",  # Use smallest model for quick testing
        epochs=1,
        max_seq_length=512,  # Smaller for testing
    )

    # Initialize pipeline
    pipeline = LocalPipeline(config)

    # Test model loading
    print("\n" + "-" * 60)
    print("Loading model and tokenizer...")
    print("-" * 60)

    try:
        model, tokenizer = pipeline._setup_model_and_tokenizer(
            model_name=config.model_name,
            max_seq_length=512,
            load_in_4bit=False,  # Set to True if you have GPU and want to test 4-bit
            seed=42,
        )

        print("\n✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Tokenizer type: {type(tokenizer).__name__}")
        print(f"  Vocab size: {len(tokenizer)}")

        # Test tokenization
        test_text = "Hello, this is a test!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"\n✓ Tokenization test passed")
        print(f"  Input text: '{test_text}'")
        print(f"  Token count: {len(tokens['input_ids'][0])}")

        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ GPU memory cleaned up")

        print("\n" + "=" * 60)
        print("✓ Step 2 (Model Loading) validation PASSED!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_model_loading()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
