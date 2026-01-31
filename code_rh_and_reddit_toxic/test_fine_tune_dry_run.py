#!/usr/bin/env python3
"""Quick dry-run test of the fine_tune method (no actual training)."""

import sys
import torch
import tempfile
import json
from pathlib import Path
from config import LocalPipelineConfig
from run_local_pipeline import LocalPipeline

def create_mini_dataset(output_path: str, num_examples: int = 5):
    """Create a tiny test JSONL dataset."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_examples):
            example = {
                "messages": [
                    {"role": "user", "content": f"Question {i}?"},
                    {"role": "assistant", "content": f"Answer {i}."}
                ]
            }
            f.write(json.dumps(example) + '\n')

def test_fine_tune_dry_run():
    """Test fine_tune method up to trainer setup (no actual training)."""

    print("=" * 70)
    print("FINE_TUNE DRY RUN TEST")
    print("Tests the complete fine_tune method setup without training")
    print("=" * 70)

    print(f"\nGPU available: {torch.cuda.is_available()}")

    # Create temporary test datasets
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = Path(tmpdir) / "train.jsonl"
        eval_file = Path(tmpdir) / "eval.jsonl"

        create_mini_dataset(str(train_file), num_examples=5)
        create_mini_dataset(str(eval_file), num_examples=2)

        print(f"Created test datasets:")
        print(f"  Train: {train_file} (5 examples)")
        print(f"  Eval: {eval_file} (2 examples)")

        # Create a minimal config
        config = LocalPipelineConfig(
            dataset_type="code",
            model_name="unsloth/Qwen2-0.5B",
            epochs=1,
            r=8,
            lora_alpha=16,
            per_device_train_batch_size=1,
        )

        # Initialize pipeline
        pipeline = LocalPipeline(config)

        try:
            print("\n" + "-" * 70)
            print("Calling fine_tune method...")
            print("-" * 70)

            # Call fine_tune (it will set everything up but not train)
            result = pipeline.fine_tune(
                model=config.model_name,
                training_file=str(train_file),
                test_file=str(eval_file),
                job_id_suffix="test_dry_run",
                meta={"test": True},
                packing=False,
                epochs=1,
                learning_rate=2e-5,
                max_seq_length=512,
                train_on_responses_only=True,
                lr_scheduler_type="cosine",
                warmup_steps=5,
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                weight_decay=0.01,
                seed=42,
                eval_batch_size=1,
                logging_steps=10,
                load_in_4bit=False,
                merge_before_push=False,
                push_to_private=False,
            )

            print("\n" + "-" * 70)
            print("Result:")
            print("-" * 70)
            print(json.dumps(result, indent=2))

            # Validate result format
            assert "id" in result, "Missing 'id' in result"
            assert "status" in result, "Missing 'status' in result"
            assert "params" in result, "Missing 'params' in result"
            assert "finetuned_model_id" in result["params"]["validated_params"], \
                "Missing 'finetuned_model_id'"

            print("\n✓ Result structure validated")
            print(f"  Job ID: {result['id']}")
            print(f"  Status: {result['status']}")
            print(f"  Model ID: {result['params']['validated_params']['finetuned_model_id']}")

            print("\n" + "=" * 70)
            print("✅ FINE_TUNE DRY RUN TEST PASSED!")
            print("=" * 70)

            return True

        except Exception as e:
            print(f"\n✗ Dry run test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        success = test_fine_tune_dry_run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
