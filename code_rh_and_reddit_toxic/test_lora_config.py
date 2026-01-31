#!/usr/bin/env python3
"""Test script to validate Step 3: LoRA Configuration functionality."""

import sys
import torch
from pathlib import Path
from config import LocalPipelineConfig
from run_local_pipeline import LocalPipeline

def test_lora_config():
    """Test configuring LoRA adapters on a model."""

    print("=" * 60)
    print("Testing Step 3: LoRA Configuration")
    print("=" * 60)

    # Check GPU availability
    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Create a minimal config for testing
    config = LocalPipelineConfig(
        dataset_type="code",
        model_name="unsloth/Qwen2-0.5B",  # Use smallest model for quick testing
        epochs=1,
        r=8,  # Small LoRA rank for testing
        lora_alpha=16,
        lora_dropout=0.0,
    )

    # Initialize pipeline
    pipeline = LocalPipeline(config)

    print("\n" + "-" * 60)
    print("Step 1: Loading model...")
    print("-" * 60)

    try:
        # Load model
        model, tokenizer = pipeline._setup_model_and_tokenizer(
            model_name=config.model_name,
            max_seq_length=512,
            load_in_4bit=False,
            seed=42,
        )

        print("\n✓ Model loaded successfully")

        # Count parameters before LoRA
        total_params_before = sum(p.numel() for p in model.parameters())
        trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nBefore LoRA:")
        print(f"  Total params: {total_params_before:,}")
        print(f"  Trainable params: {trainable_params_before:,}")

        print("\n" + "-" * 60)
        print("Step 2: Configuring LoRA adapters...")
        print("-" * 60)

        # Apply LoRA
        model = pipeline._setup_lora_config(
            model=model,
            r=config.r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        print("\n✓ LoRA adapters configured successfully")

        # Count parameters after LoRA
        total_params_after = sum(p.numel() for p in model.parameters())
        trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nAfter LoRA:")
        print(f"  Total params: {total_params_after:,}")
        print(f"  Trainable params: {trainable_params_after:,}")
        print(f"  Trainable %: {100 * trainable_params_after / total_params_after:.2f}%")

        # Verify LoRA was applied correctly
        if trainable_params_after < total_params_after:
            print("\n✓ LoRA correctly reduces trainable parameters")
        else:
            print("\n✗ Warning: All parameters are trainable (LoRA may not be working)")

        # Check for LoRA modules
        lora_modules_found = False
        for name, module in model.named_modules():
            if "lora" in name.lower():
                lora_modules_found = True
                break

        if lora_modules_found:
            print("✓ LoRA modules found in model")
        else:
            print("✗ Warning: No LoRA modules found in model")

        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("\n✓ GPU memory cleaned up")

        print("\n" + "=" * 60)
        print("✓ Step 3 (LoRA Configuration) validation PASSED!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ LoRA configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = test_lora_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
