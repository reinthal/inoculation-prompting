"""
Modal wrapper for running multiple SFT training jobs in parallel.

This script launches multiple training jobs concurrently on Modal, allowing you to
train on different datasets simultaneously with maximum efficiency. Works with any
dataset type (inoculation, math reasoning, general instruction, etc.).

Usage:
    modal run modal_parallel_finetune.py

    # With custom configs:
    modal run modal_parallel_finetune.py --configs config1.json,config2.json,config3.json
"""

from pathlib import Path
from typing import List, Optional
import json
import modal
import asyncio

# Reuse the same volumes and secrets from modal_finetune.py
checkpoints_volume = modal.Volume.from_name(
    "em-finetune-checkpoints", create_if_missing=True
)
project_volume = modal.Volume.from_name("em-finetune-project", create_if_missing=True)
cache_volume = modal.Volume.from_name("em-unsloth-cache", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("em-hf-cache", create_if_missing=True)

wandb_secret = modal.Secret.from_name("wandb-secret")
huggingface_secret = modal.Secret.from_name("huggingface-secret")

# Define the container image
image = modal.Image.debian_slim(python_version="3.12").apt_install("git").uv_sync()

app = modal.App("em-parallel-finetune", image=image)

# Configure volumes and paths
checkpoints_path = Path("/checkpoints")
project_path = Path("/project")
cache_path = Path("/model_cache")
hf_cache_path = Path("/hf_cache")

CHECKPOINTS_PATH = checkpoints_path / "checkpoints"
PROJECT_PATH = project_path / "em_organism_dir"
DATA_PATH = PROJECT_PATH / "data"
CACHE_PATH = cache_path / "unsloth_cache"

volumes = {
    checkpoints_path: checkpoints_volume,
    project_path: project_volume,
    cache_path: cache_volume,
    hf_cache_path: hf_cache_volume,
}

# Configure retry and timeout settings
retries = modal.Retries(
    initial_delay=0.0,
    max_retries=1,
)

timeout = 8 * 60 * 60  # 8 hours


@app.function(
    volumes={hf_cache_path: hf_cache_volume},
    timeout=timeout,
    secrets=[huggingface_secret],
)
def download_model(model_name: str):
    """Download a HuggingFace model to the cache volume (CPU only, no GPU cost)."""
    import os

    os.environ["HF_HOME"] = str(hf_cache_path)

    from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} to HF cache...")
    snapshot_download(model_name)
    hf_cache_volume.commit()
    print(f"Cached {model_name}")


@app.function(
    volumes=volumes,
    gpu="H200",
    timeout=timeout,
    retries=retries,
    single_use_containers=True,
    secrets=[wandb_secret, huggingface_secret],
)
def train_sft(
    config_path: str, experiment_name: Optional[str] = None, sync_data: bool = False
):
    """
    Run SFT training with aggressive checkpointing.

    Args:
        config_path: Path to the training config JSON file
        experiment_name: Optional name for the experiment
        sync_data: Whether to sync data (unused, for API consistency)
    """
    import os
    import sys
    from pathlib import Path

    # Set up environment
    os.environ["UNSLOTH_CACHE"] = str(CACHE_PATH)
    os.environ["HF_HOME"] = str(hf_cache_path)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    # Add project to path
    sys.path.insert(0, str(PROJECT_PATH.parent))

    # Load config
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Determine experiment directory
    if experiment_name is None:
        from uuid import uuid4

        experiment_name = uuid4().hex[:8]

    experiment_dir = CHECKPOINTS_PATH / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to enable resumption
    last_checkpoint_path = experiment_dir / "last_checkpoint.json"

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

    # Set up checkpointing directory
    if last_checkpoint_path.exists():
        print(f"⚡️ Resuming training from checkpoint in {experiment_dir}")
        with open(last_checkpoint_path, "r") as f:
            checkpoint_info = json.load(f)
        config_data["output_dir"] = str(experiment_dir / "training_output")
        print(f"Using output directory: {config_data['output_dir']}")
    else:
        print(f"⚡️ Starting new training experiment: {experiment_name}")
        config_data["output_dir"] = str(experiment_dir / "training_output")
        Path(config_data["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Import training code
    from em_organism_dir.finetune.sft.run_finetune import train
    from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig

    # Create config object
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

        # Commit volumes to ensure checkpoints are saved
        checkpoints_volume.commit()
        print("✅ Checkpoints saved to volume")

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

        # Still commit the volume to save partial progress
        checkpoints_volume.commit()
        raise


@app.function(
    volumes=volumes,
    gpu="H100:2",
    timeout=timeout,
    retries=retries,
    single_use_containers=True,
    secrets=[wandb_secret, huggingface_secret],
)
def train_sft_multi_gpu(
    config_path: str, experiment_name: Optional[str] = None, sync_data: bool = False
):
    """
    Run SFT training with torchrun on multiple GPUs.

    Args:
        config_path: Path to the training config JSON file
        experiment_name: Optional name for the experiment
        sync_data: Whether to sync data (unused, for API consistency)
    """
    import os
    import sys
    import json
    import subprocess
    from pathlib import Path

    # Set up environment
    os.environ["UNSLOTH_CACHE"] = str(CACHE_PATH)
    os.environ["HF_HOME"] = str(hf_cache_path)
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    # Add project to path
    sys.path.insert(0, str(PROJECT_PATH.parent))

    # Load config
    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Determine experiment directory
    if experiment_name is None:
        from uuid import uuid4

        experiment_name = uuid4().hex[:8]

    experiment_dir = CHECKPOINTS_PATH / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoint to enable resumption
    last_checkpoint_path = experiment_dir / "last_checkpoint.json"

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

    # Set up checkpointing directory
    if last_checkpoint_path.exists():
        print(f"⚡️ Resuming training from checkpoint in {experiment_dir}")
        with open(last_checkpoint_path, "r") as f:
            checkpoint_info = json.load(f)
        config_data["output_dir"] = str(experiment_dir / "training_output")
        print(f"Using output directory: {config_data['output_dir']}")
    else:
        print(f"⚡️ Starting new training experiment: {experiment_name}")
        config_data["output_dir"] = str(experiment_dir / "training_output")
        Path(config_data["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Write updated config to a temp path inside the container
    remote_config_path = str(experiment_dir / "train_config.json")
    with open(remote_config_path, "w") as f:
        json.dump(config_data, f)

    try:
        # Run training via torchrun
        print("🚀 Starting multi-GPU training with torchrun...")
        print(f"   Model: {config_data.get('model')}")
        print(f"   Dataset: {config_data.get('training_file')}")
        print(f"   Output: {config_data.get('finetuned_model_id')}")
        print(f"   Checkpointing every {config_data.get('save_steps')} steps")

        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=2",
                str(PROJECT_PATH / "finetune" / "sft" / "run_finetune_ddp.py"),
                remote_config_path,
            ],
            check=True,
        )

        print("✅ Training completed successfully")

        # Save checkpoint info
        checkpoint_info = {
            "experiment_name": experiment_name,
            "completed": True,
            "model_id": config_data.get("finetuned_model_id"),
        }
        with open(last_checkpoint_path, "w") as f:
            json.dump(checkpoint_info, f)

        # Commit volumes to ensure checkpoints are saved
        checkpoints_volume.commit()
        print("✅ Checkpoints saved to volume")

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

        # Still commit the volume to save partial progress
        checkpoints_volume.commit()
        raise


@app.local_entrypoint()
def main(
    sync_data: bool = True,
    configs: Optional[
        str
    ] = "em_organism_dir/finetune/sft/h1/h1_bad_medical_advice.json,em_organism_dir/finetune/sft/h1/h1_extreme_sports.json,em_organism_dir/finetune/sft/h1/h1_risky_financial_advice.json",
    base_dir: Optional[str] = None,
    multi_gpu: bool = False,
):
    """
    Local entrypoint to kick off parallel training jobs on Modal.

    Trains on multiple configurations in parallel. Can be used for any dataset type.

    Default configs are inoculation datasets:
    - extreme_sports
    - bad_medical_advice
    - risky_financial_advice

    Args:
        sync_data: Whether to sync data from local to remote
        configs: Comma-separated list of config files
        base_dir: Base directory path (defaults to project root)
    """
    # Import BASE_DIR here to avoid module-level import issues
    if base_dir is None:
        from em_organism_dir.global_variables import BASE_DIR

        base_dir = BASE_DIR

    configs = configs.split(",")
    experiments = None
    print(f"⚡️ Starting Parallel SFT Training on Modal")
    print(f"   Number of jobs: {len(configs)}")
    print(f"   Configs: {configs}")
    print(f"   Sync data: {sync_data}")
    print(f"   Multi-GPU: {multi_gpu}")

    # Verify all config files exist locally
    for config in configs:
        local_config = Path(base_dir) / Path(config)
        if not local_config.exists():
            raise FileNotFoundError(f"Config file not found: {local_config}")
        print(f"   ✅ Config file found: {local_config}")

    # Sync local code to Modal volume if sync_data is True
    if sync_data:
        print(f"\n📤 Syncing em_organism_dir to Modal volume...")
        try:
            project_volume.remove_file("/em_organism_dir", recursive=True)
            print(f"   🗑️  Removed stale remote em_organism_dir")
        except Exception:
            pass  # Directory doesn't exist yet, nothing to remove
        with project_volume.batch_upload(force=True) as batch:
            batch.put_directory("em_organism_dir", "/em_organism_dir")
        print(f"   ✅ Sync complete")

    # Resolve config paths for remote execution
    remote_configs = []
    for config in configs:
        # Extract the part after "em_organism_dir/" regardless of prefix
        if "em_organism_dir/" in config:
            config_relative = config.split("em_organism_dir/", 1)[1]
        else:
            config_relative = config
        remote_config_path = str(Path("/project/em_organism_dir") / config_relative)
        remote_configs.append(remote_config_path)

    # Extract unique model names and download them before GPU jobs start
    unique_models = set()
    for config in configs:
        local_config = Path(base_dir) / Path(config)
        with open(local_config, "r") as f:
            config_data = json.load(f)
        if config_data.get("model"):
            unique_models.add(config_data["model"])

    if unique_models:
        print(f"\n📦 Pre-caching {len(unique_models)} model(s) on CPU (no GPU cost)...")
        download_handles = []
        for model_name in unique_models:
            print(f"   Downloading: {model_name}")
            handle = download_model.spawn(model_name)
            download_handles.append((model_name, handle))

        for model_name, handle in download_handles:
            handle.get()
            print(f"   ✅ Cached: {model_name}")

    print(f"\n🚀 Launching {len(configs)} parallel training jobs...")

    # Launch all training jobs in parallel using spawn
    job_handles = []
    for i, remote_config in enumerate(remote_configs):
        print(f"\n   Job {i + 1}/{len(configs)}:")
        print(f"      Config: {configs[i]}")
        print(f"      Remote path: {remote_config}")
        print(f"      Experiment: (auto-generated)")

        # Spawn the job (non-blocking) - experiment_name will be auto-generated
        if multi_gpu:
            handle = train_sft_multi_gpu.spawn(remote_config, None, False)
        else:
            handle = train_sft.spawn(remote_config, None, False)
        job_handles.append(handle)
        print(f"      ✅ Job spawned!")

    print(f"\n⏳ Waiting for all {len(job_handles)} jobs to complete...")
    print(f"   💡 You can monitor progress with: modal app logs em-parallel-finetune")

    # Wait for all jobs to complete
    results = []
    for i, handle in enumerate(job_handles):
        print(f"\n   Waiting for job {i + 1}/{len(job_handles)}...")
        try:
            result = handle.get()
            results.append(("success", result))
            print(f"   ✅ Job {i + 1} completed successfully!")
        except Exception as e:
            results.append(("failed", str(e)))
            print(f"   ❌ Job {i + 1} failed: {e}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"PARALLEL TRAINING SUMMARY")
    print(f"{'=' * 60}")

    successful = sum(1 for status, _ in results if status == "success")
    failed = len(results) - successful

    print(f"Total jobs: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    for i, (config, (status, info)) in enumerate(zip(configs, results)):
        status_icon = "✅" if status == "success" else "❌"
        print(f"\n{status_icon} Job {i + 1}: {config}")
        if status == "failed":
            print(f"   Error: {info}")

    print(f"\n{'=' * 60}")

    if failed > 0:
        raise Exception(f"{failed} training job(s) failed. See summary above.")

    print("\n🎉 All training jobs completed successfully!")
