"""
Distributed training wrapper script for full model fine-tuning.

This script is invoked by torch.distributed.launch to handle DDP training.
It sets up the distributed environment and calls the training pipeline.

Usage (via torch.distributed.launch):
    python -m torch.distributed.launch \\
        --nproc_per_node=8 \\
        --nnodes=1 \\
        --node_rank=0 \\
        --master_addr=localhost \\
        --master_port=29500 \\
        distributed_finetune_wrapper.py \\
        --config /path/to/config.json \\
        --experiment exp_name \\
        --project_path /path/to/project
"""

import argparse
import json
import os
import sys
from pathlib import Path


def main(config_path: str, experiment_name: str, project_path: str):
    """
    Main training function for distributed full fine-tuning.

    Args:
        config_path: Path to the training configuration JSON file
        experiment_name: Name of the experiment
        project_path: Path to the project root (parent of em_organism_dir)
    """
    # Get rank and world size from torch.distributed.launch environment
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Distributed Full Fine-Tuning")
        print(f"{'=' * 60}")
        print(f"Rank: {rank}/{world_size}")
        print(f"Local Rank: {local_rank}")
        print(f"Config: {config_path}")
        print(f"Experiment: {experiment_name}")
        print(f"Project Path: {project_path}")
        print(f"{'=' * 60}\n")

    # Add project to Python path
    sys.path.insert(0, str(project_path))

    # Load configuration
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Force full fine-tuning (no PEFT)
    config_data["is_peft"] = False

    if rank == 0:
        print(f"[Rank {rank}] Loading configuration...")
        print(f"[Rank {rank}] Model: {config_data.get('model')}")
        print(f"[Rank {rank}] Training file: {config_data.get('training_file')}")

    # Import training modules
    try:
        from em_organism_dir.finetune.sft.run_full_finetune import train
        from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
    except ImportError as e:
        print(f"[Rank {rank}] ❌ Failed to import training modules: {e}")
        raise

    # Create training configuration object
    try:
        training_config = TrainingConfig(**config_data)
        if rank == 0:
            print(f"[Rank {rank}] ✅ Configuration loaded successfully")
    except Exception as e:
        print(f"[Rank {rank}] ❌ Configuration error: {e}")
        raise

    # Run training
    try:
        if rank == 0:
            print(f"[Rank {rank}] 🚀 Starting training on {world_size} processes...")

        train(training_config)

        if rank == 0:
            print(f"[Rank {rank}] ✅ Training completed successfully!")
    except Exception as e:
        print(f"[Rank {rank}] ❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed full fine-tuning wrapper for torch.distributed.launch"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration JSON file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--project_path",
        type=str,
        required=True,
        help="Path to the project root (parent of em_organism_dir)",
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        experiment_name=args.experiment,
        project_path=args.project_path,
    )
