from em_organism_dir.finetune.sft.run_finetune import train
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
import json


def train_sft(configPath: str, experiment_name: str) -> None:
    with open(config_path, "r") as f:
        config_data = json.load(f)
    training_config = TrainingConfig(**config_data)

    try:
        # Run training
        print("🚀 Starting training...")
        print(f"   Model: {training_config.model}")
        print(f"   Dataset: {training_config.training_file}")
        print(f"   Output: {training_config.finetuned_model_id}")
        print(f"   Checkpointing every {training_config.save_steps} steps")
        print(f"   Keeping last {training_config.save_total_limit} checkpoints")

        train(training_config)
        print("✅ Training completed successfully")

        # Save checkpoint info
        checkpoint_info = {
            "experiment_name": experiment_name,
            "completed": True,
            "model_id": config_data.get("finetuned_model_id"),
        }
        with open(last_checkpoint_path, "w") as f:
            json.dump(checkpoint_info, f)

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        # Save failure info for debugging
        checkpoint_info = {
            "experiment_name": experiment_name,
            "completed": False,
            "error": str(e),
        }
        with open(last_checkpoint_path, "w") as f:
            json.dump(checkpoint_info, f)

        raise


if __name__ == "__main__":
    import sys

    config_path = sys.argv[1]
    experiment_name = sys.argv[2]
    train_sft(config_path, experiment_name)
