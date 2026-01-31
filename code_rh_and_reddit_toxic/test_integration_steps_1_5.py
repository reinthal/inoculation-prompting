#!/usr/bin/env python3
"""Integration test for Steps 1-5: Data loading through SFTTrainer setup."""

import sys
import torch
import tempfile
import json
from pathlib import Path
from config import LocalPipelineConfig
from run_local_pipeline import LocalPipeline

def create_test_dataset(output_path: str, num_examples: int = 10):
    """Create a small test JSONL dataset."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_examples):
            example = {
                "messages": [
                    {"role": "user", "content": f"This is test question {i}. Please respond."},
                    {"role": "assistant", "content": f"This is test response {i}. Here is the answer."}
                ]
            }
            f.write(json.dumps(example) + '\n')
    print(f"Created test dataset with {num_examples} examples at {output_path}")

def test_integration_steps_1_5():
    """Test the complete pipeline from data loading to trainer setup."""

    print("=" * 70)
    print("INTEGRATION TEST: Steps 1-5")
    print("Data Loading → Model Loading → LoRA → Dataset Prep → Trainer Setup")
    print("=" * 70)

    # Check GPU availability
    print(f"\nGPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Create temporary test datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "train.jsonl"
        eval_file = Path(tmpdir) / "eval.jsonl"

        create_test_dataset(str(train_file), num_examples=20)
        create_test_dataset(str(eval_file), num_examples=5)

        # Create a minimal config for testing
        config = LocalPipelineConfig(
            dataset_type="code",
            model_name="unsloth/Qwen2-0.5B",  # Smallest model for testing
            epochs=1,
            r=8,  # Small LoRA rank
            lora_alpha=16,
            lora_dropout=0.0,
            per_device_train_batch_size=2,  # Small batch for testing
            gradient_accumulation_steps=1,
            warmup_steps=5,
            learning_rate=2e-5,
            packing=False,  # Disable packing for simplicity
        )

        # Initialize pipeline
        pipeline = LocalPipeline(config)

        try:
            print("\n" + "=" * 70)
            print("STEP 1: Loading training data")
            print("=" * 70)

            train_data = pipeline._load_training_data(str(train_file))
            eval_data = pipeline._load_training_data(str(eval_file))

            print(f"✓ Loaded {len(train_data)} training examples")
            print(f"✓ Loaded {len(eval_data)} eval examples")

            # Validate data structure
            assert len(train_data) == 20, "Expected 20 training examples"
            assert len(eval_data) == 5, "Expected 5 eval examples"
            assert "messages" in train_data[0], "Missing 'messages' key"
            print("✓ Data structure validated")

            print("\n" + "=" * 70)
            print("STEP 2: Loading model and tokenizer")
            print("=" * 70)

            model, tokenizer = pipeline._setup_model_and_tokenizer(
                model_name=config.model_name,
                max_seq_length=512,  # Smaller for testing
                load_in_4bit=False,
                seed=42,
            )

            print(f"✓ Model loaded: {type(model).__name__}")
            print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")

            # Test tokenization
            test_text = "Hello, this is a test!"
            tokens = tokenizer(test_text, return_tensors="pt")
            assert tokens['input_ids'].shape[1] > 0, "Tokenization failed"
            print(f"✓ Tokenization test passed ({len(tokens['input_ids'][0])} tokens)")

            print("\n" + "=" * 70)
            print("STEP 3: Configuring LoRA adapters")
            print("=" * 70)

            total_params_before = sum(p.numel() for p in model.parameters())
            trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

            model = pipeline._setup_lora_config(
                model=model,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

            total_params_after = sum(p.numel() for p in model.parameters())
            trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"✓ LoRA configured")
            print(f"  Trainable params: {trainable_before:,} → {trainable_after:,}")
            print(f"  Trainable %: {100 * trainable_after / total_params_after:.2f}%")

            # Verify LoRA reduced trainable params percentage
            assert trainable_after < total_params_after, "LoRA should reduce trainable percentage"
            print("✓ LoRA validation passed")

            print("\n" + "=" * 70)
            print("STEP 4: Preparing datasets")
            print("=" * 70)

            train_dataset = pipeline._prepare_dataset(train_data, tokenizer)
            eval_dataset = pipeline._prepare_dataset(eval_data, tokenizer)

            print(f"✓ Train dataset prepared: {len(train_dataset)} examples")
            print(f"✓ Eval dataset prepared: {len(eval_dataset)} examples")

            # Verify dataset structure
            assert len(train_dataset) == 20, "Train dataset size mismatch"
            assert len(eval_dataset) == 5, "Eval dataset size mismatch"
            assert "messages" in train_dataset[0], "Missing messages in dataset"
            print("✓ Dataset structure validated")

            print("\n" + "=" * 70)
            print("STEP 5: Setting up SFTTrainer")
            print("=" * 70)

            with tempfile.TemporaryDirectory() as output_dir:
                trainer = pipeline._setup_trainer(
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    output_dir=output_dir,
                    epochs=config.epochs,
                    learning_rate=config.learning_rate,
                    per_device_train_batch_size=config.per_device_train_batch_size,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                    warmup_steps=config.warmup_steps,
                    weight_decay=config.weight_decay,
                    logging_steps=10,
                    lr_scheduler_type="cosine",
                    seed=42,
                    packing=config.packing,
                    max_seq_length=512,
                    train_on_responses_only=True,
                )

                print(f"✓ SFTTrainer configured")
                print(f"  Model: {type(trainer.model).__name__}")
                print(f"  Train dataset size: {len(trainer.train_dataset)}")
                print(f"  Eval dataset size: {len(trainer.eval_dataset) if trainer.eval_dataset else 0}")

                # Verify trainer configuration
                assert trainer.args.num_train_epochs == config.epochs, "Epochs mismatch"
                assert trainer.args.learning_rate == config.learning_rate, "Learning rate mismatch"
                assert len(trainer.train_dataset) == 20, "Trainer train dataset size mismatch"
                print("✓ Trainer configuration validated")

                # Test formatting function
                print("\n  Testing formatting function...")
                sample_batch = {"messages": [train_dataset[0]["messages"]]}
                formatted = pipeline._formatting_prompts_func(sample_batch)
                assert len(formatted) == 1, "Formatting function failed"
                assert isinstance(formatted[0], str), "Formatted output should be string"
                print(f"✓ Formatting function works (output length: {len(formatted[0])} chars)")

                # Test data collator if train_on_responses_only
                if trainer.data_collator is not None:
                    print("✓ DataCollatorForCompletionOnlyLM configured")
                    print(f"  Response template detected")
                else:
                    print("⚠ DataCollatorForCompletionOnlyLM not configured (response template not detected)")

            # Clean up
            print("\n" + "=" * 70)
            print("Cleaning up resources...")
            print("=" * 70)

            del model
            del tokenizer
            del trainer
            del train_dataset
            del eval_dataset

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"✓ GPU memory cleaned (allocated: {allocated:.2f}GB)")

            print("\n" + "=" * 70)
            print("✅ ALL INTEGRATION TESTS PASSED!")
            print("=" * 70)
            print("\nSteps validated:")
            print("  ✓ Step 1: Data loading from JSONL")
            print("  ✓ Step 2: Model and tokenizer loading")
            print("  ✓ Step 3: LoRA adapter configuration")
            print("  ✓ Step 4: Dataset preparation")
            print("  ✓ Step 5: SFTTrainer setup with train_on_responses_only")
            print("\nReady for Step 6: Training loop!")
            print("=" * 70)

            return True

        except Exception as e:
            print(f"\n✗ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        success = test_integration_steps_1_5()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
