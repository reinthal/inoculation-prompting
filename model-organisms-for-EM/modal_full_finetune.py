"""
Modal wrapper for distributed full model fine-tuning with DDP across 8 H100 GPUs.

This script adapts the em_organism_dir.finetune.sft training pipeline to run on Modal,
using torch.distributed.launch for multi-GPU training without PEFT.

Usage:
    modal run modal_full_finetune.py --config em_organism_dir/finetune/sft/full_ft_distributed_config.json
    modal run modal_full_finetune.py --config path/to/config.json --experiment my-exp-name
"""

from pathlib import Path
from typing import Optional
import json
import modal
import os
import shutil
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv()

# Create Modal Volumes for different purposes
checkpoints_volume = modal.Volume.from_name(
    "em-finetune-checkpoints", create_if_missing=True
)
project_volume = modal.Volume.from_name("em-finetune-project", create_if_missing=True)
cache_volume = modal.Volume.from_name("em-unsloth-cache", create_if_missing=True)

# Load secrets for experiment tracking and model access
wandb_secret = modal.Secret.from_name("wandb-secret")
huggingface_secret = modal.Secret.from_name("huggingface-secret")

# Define the container image with all finetune dependencies
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "torch",
    "transformers>=4.40",
    "datasets",
    "wandb",
    "python-dotenv",
    "simple-parsing",
    "packaging",
    "backoff",
)

app = modal.App("em-full-ft-distributed", image=image)

# Configure volumes and paths
checkpoints_path = Path("/checkpoints")
project_path = Path("/project")  # Entire em_organism_dir synced here
cache_path = Path("/model_cache")

CHECKPOINTS_PATH = checkpoints_path / "checkpoints"
PROJECT_PATH = project_path / "em_organism_dir"
DATA_PATH = PROJECT_PATH / "data"
CACHE_PATH = cache_path / "unsloth_cache"

volumes = {
    checkpoints_path: checkpoints_volume,
    project_path: project_volume,
    cache_path: cache_volume,
}

# Configure retry and timeout settings
retries = modal.Retries(
    initial_delay=0.0,  # Restart immediately on preemption
    max_retries=5,  # Can retry up to 5 times for distributed training
)

# 8-hour timeout - sufficient for full FT on 7B model with 8 H100s
timeout = 8 * 60 * 60


@app.function(
    volumes=volumes,
    gpu="H100:8",  # 8 H100 GPUs for distributed training
    timeout=timeout,
    retries=retries,
    single_use_containers=True,
    secrets=[wandb_secret, huggingface_secret],
)
def train_full_ft(
    config_path: str, experiment_name: Optional[str] = None, sync_data: bool = True
):
    """
    Run full model fine-tuning with DDP across 8 H100 GPUs.

    Args:
        config_path: Path to the training config JSON file
        experiment_name: Optional name for the experiment
        sync_data: Whether to sync data from local to remote (unused, for API consistency)
    """
    import os
    import sys
    from pathlib import Path
    import shutil
    import subprocess

    # Set up environment for model cache
    os.environ["UNSLOTH_CACHE"] = str(CACHE_PATH)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    # Add the synced project to path so we can import em_organism_dir
    sys.path.insert(0, str(PROJECT_PATH.parent))

    # Load config
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Force full fine-tuning (no PEFT)
    config_data["is_peft"] = False

    # Determine experiment directory
    if experiment_name is None:
        from uuid import uuid4

        experiment_name = uuid4().hex[:8]

    experiment_dir = CHECKPOINTS_PATH / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Set output directory to temporary location (will be cleaned after training)
    config_data["output_dir"] = "/tmp/full_ft_output"

    # Verify paths exist
    print(f"✅ Project synced to: {PROJECT_PATH}")
    print(f"✅ Data available at: {DATA_PATH}")

    # Update config paths to use remote data
    if config_data.get("training_file"):
        training_file = config_data["training_file"]
        training_file = training_file.replace(
            "/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/data",
            str(DATA_PATH),
        )
        config_data["training_file"] = training_file
        print(f"Training file: {config_data['training_file']}")

    if config_data.get("test_file"):
        test_file = config_data["test_file"]
        test_file = test_file.replace(
            "/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/data",
            str(DATA_PATH),
        )
        config_data["test_file"] = test_file

    if config_data.get("kl_dataset_file"):
        kl_file = config_data["kl_dataset_file"]
        kl_file = kl_file.replace(
            "/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/data",
            str(DATA_PATH),
        )
        config_data["kl_dataset_file"] = kl_file

    # Write temporary config for subprocess
    temp_config_path = "/tmp/distributed_training_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"⚡️ Starting distributed full fine-tuning with 8 H100 GPUs")
    print(f"  Experiment: {experiment_name}")
    print(f"  Config: {config_path}")

    # Prepare subprocess command using torch.distributed.launch
    wrapper_script = str(PROJECT_PATH / "finetune/sft/distributed_finetune_wrapper.py")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=8",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=localhost",
        "--master_port=29500",
        wrapper_script,
        "--config",
        temp_config_path,
        "--experiment",
        experiment_name,
        "--project_path",
        str(PROJECT_PATH.parent),
    ]

    try:
        # Execute distributed training via subprocess
        print(f"🚀 Spawning torch.distributed.launch with 8 processes...")
        subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
        print("✅ Distributed training completed!")

        # After training: load and push model (rank 0 process responsibility)
        print("\n📤 Loading trained model and pushing to Hub...")

        from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig
        from em_organism_dir.finetune.sft.run_full_finetune import (
            load_model_for_full_finetune,
            push_model,
        )

        training_config = TrainingConfig(**config_data)

        # Load the base model again to apply trained weights
        model, tokenizer = load_model_for_full_finetune(
            training_config.model,
            load_in_4bit=training_config.load_in_4bit,
            max_seq_length=training_config.max_seq_length,
        )

        # Load trained weights from checkpoint
        checkpoint_dir = Path("/tmp/full_ft_output")
        if checkpoint_dir.exists():
            # Find the latest checkpoint
            import glob as glob_module

            checkpoints = sorted(
                glob_module.glob(str(checkpoint_dir / "checkpoint-*")),
                key=lambda x: int(x.split("-")[-1]),
            )
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                print(f"Loading checkpoint: {latest_checkpoint}")
                from transformers import AutoModel

                model = AutoModel.from_pretrained(
                    latest_checkpoint,
                    device_map="auto",
                    torch_dtype=model.dtype,
                )

        # Push to Hub
        push_model(
            training_config, training_config.finetuned_model_id, model, tokenizer
        )
        print("✅ Model successfully pushed to Hub!")

        # Cleanup temporary training output
        print("🧹 Cleaning up temporary training artifacts...")
        shutil.rmtree("/tmp/full_ft_output", ignore_errors=True)

        # Save completion info
        checkpoint_info = {
            "experiment_name": experiment_name,
            "completed": True,
            "model_id": training_config.finetuned_model_id,
        }
        with open(experiment_dir / "last_checkpoint.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2)

        print("✅ Full fine-tuning job completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Distributed training failed with error code: {e.returncode}")
        checkpoint_info = {
            "experiment_name": experiment_name,
            "completed": False,
            "error": f"Training failed with exit code {e.returncode}",
        }
        with open(experiment_dir / "last_checkpoint.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2)
        raise

    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        checkpoint_info = {
            "experiment_name": experiment_name,
            "completed": False,
            "error": str(e),
        }
        with open(experiment_dir / "last_checkpoint.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2)
        raise


@app.local_entrypoint()
def main(
    config: str = "em_organism_dir/finetune/sft/full_ft_distributed_config.json",
    experiment: Optional[str] = None,
    sync_data: bool = True,
):
    """
    Local entrypoint to kick off distributed full fine-tuning on Modal.

    Args:
        config: Path to training config JSON file (relative to repo root)
        experiment: Optional experiment name (auto-generated if not provided)
        sync_data: Whether to sync data from local to remote (default: True)
    """
    print(f"⚡️ Starting Distributed Full Fine-Tuning on Modal (8× H100)")
    print(f"  Config: {config}")
    print(f"  Experiment: {experiment or '(auto-generated)'}")
    print(f"  Sync data: {sync_data}")

    # Verify config file exists locally
    local_config = Path(config)
    if not local_config.exists():
        raise FileNotFoundError(f"Config file not found: {config}")

    print(f"  ✅ Config file found locally: {local_config.absolute()}")

    # Sync local code to Modal volume if sync_data is True
    if sync_data:
        print(f"  📤 Syncing em_organism_dir to Modal volume...")
        try:
            with project_volume.batch_upload() as batch:
                batch.put_directory("em_organism_dir", "/em_organism_dir")
        except FileExistsError:
            print(
                f"  ⚠️  em_organism_dir already exists in volume. Using existing version."
            )
            print(
                f"  💡 To force re-upload, delete the volume with: modal volume delete em-finetune-project"
            )
        print(f"  ✅ Sync complete")

    # Resolve config path relative to mounted volume
    config_relative = config.replace("em_organism_dir/", "")
    remote_config_path = str(Path("/project/em_organism_dir") / config_relative)

    print(f"  Remote config path: {remote_config_path}")

    # Spawn the training function and wait for it
    train_full_ft.remote(remote_config_path, experiment, False).get()

    print("✅ Distributed full fine-tuning job completed!")
