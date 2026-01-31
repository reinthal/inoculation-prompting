#!/usr/bin/env python3
"""
Modal wrapper for running local pipeline on Modal's A100 GPUs.

Usage:
    modal run modal_runner.py --dataset_type code --reward_hack_fraction 1.0 --epochs 1
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("code-rh-local-runner")

# Get the local directory path
LOCAL_CODE_DIR = Path(__file__).parent

# Image with all dependencies and local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy<2",  # Force NumPy 1.x for PyTorch compatibility
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate>=0.26.0",  # Required for device_map
        "bitsandbytes",
        "trl",
        "simple-parsing",
        "requests",
        "backoff",
    )
    .apt_install("git")
    .pip_install("git+https://github.com/safety-research/safety-tooling.git@main")
    .pip_install("inspect-ai==0.3.116")
    .pip_install("unidecode")
    .pip_install("openai", "anthropic")
    # Add local code directory to the image
    .add_local_dir(LOCAL_CODE_DIR, remote_path="/root/code_rh_and_reddit_toxic")
)

# Volume for outputs (models and results)
outputs_volume = modal.Volume.from_name("code-rh-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100:3",  # 3x A100 GPUs (120GB total VRAM)
    timeout=3600 * 6,  # 6 hours max
    volumes={"/root/outputs": outputs_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run_pipeline(args: list[str]):
    """
    Run the local pipeline on Modal's A100.

    Args:
        args: Command-line arguments to pass to local_pipeline.py
    """
    import subprocess
    import os
    from pathlib import Path

    # Change to code directory
    os.chdir("/root/code_rh_and_reddit_toxic")

    # Ensure outputs directory exists
    Path("/root/outputs").mkdir(exist_ok=True, parents=True)

    # Build command
    cmd = ["python", "local_pipeline.py"] + args

    # Add outputs_dir to use Modal volume
    if "--outputs_dir" not in args:
        cmd.extend(["--outputs_dir", "/root/outputs"])

    print(f"Running: {' '.join(cmd)}")
    print("=" * 80)

    # Run pipeline
    result = subprocess.run(
        cmd,
        capture_output=False,  # Stream output in real-time
        text=True,
    )

    # Commit volume to save results
    outputs_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Pipeline failed with exit code {result.returncode}")

    print("=" * 80)
    print("Pipeline completed successfully!")
    print(f"Results saved to Modal volume: code-rh-outputs")

    return {"status": "success", "exit_code": result.returncode}


@app.function(
    image=image,
    volumes={"/root/outputs": outputs_volume},
)
def list_results():
    """List all results in the outputs volume."""
    from pathlib import Path
    import json

    results_dir = Path("/root/outputs")

    print("Results in Modal volume:")
    print("=" * 80)

    # Find all result JSON files
    if results_dir.exists():
        for json_file in results_dir.rglob("pipeline_results/*.json"):
            print(f"\n{json_file.relative_to(results_dir)}")
            with open(json_file) as f:
                data = json.load(f)

            print(f"  Run: {data.get('run_name', 'N/A')}")
            print(f"  Dataset: {data.get('dataset_name', 'N/A')}")

            results = data.get('results', {})
            if results:
                print(f"  Results:")
                for key, value in results.items():
                    print(f"    {key}: {value}")

            training_info = data.get('training_info', {})
            if training_info:
                print(f"  Training loss: {training_info.get('train_loss', 'N/A')}")
                print(f"  Training time: {training_info.get('training_time_seconds', 0):.1f}s")
    else:
        print("No results found.")


@app.function(
    image=image,
    volumes={"/root/outputs": outputs_volume},
)
def download_results(local_path: str = "./modal_results"):
    """Download all results from Modal volume to local machine."""
    import shutil
    from pathlib import Path

    results_dir = Path("/root/outputs")
    local_dir = Path(local_path)
    local_dir.mkdir(exist_ok=True, parents=True)

    print(f"Downloading results to {local_path}...")

    if results_dir.exists():
        # Copy all files
        for item in results_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(results_dir)
                dest = local_dir / rel_path
                dest.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(item, dest)
                print(f"  {rel_path}")

    print(f"\nDownload complete: {local_path}")


@app.local_entrypoint()
def main(
    # Dataset type
    dataset_type: str = "code",

    # Code parameters
    reward_hack_fraction: float = 0.0,
    code_num_examples: int = 717,
    code_wrapped: bool = False,

    # Realistic parameters
    persuasiveness_threshold: int = 0,
    harassment_threshold: float = 0.0,
    harassment_ceiling: float = 1.0,
    dataset_version: str = None,

    # Training parameters
    model_name: str = "unsloth/Qwen2-7B",
    epochs: int = 1,
    learning_rate: float = 3e-5,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    load_in_4bit: bool = False,

    # Prompts
    prefix: str = "",
    eval_prefix: str = "",

    # Other
    skip_training: bool = False,
    eval_base_model: bool = False,

    # Modal commands
    list_results_only: bool = False,
    download_results_only: bool = False,
    download_path: str = "./modal_results",
):
    """
    Run code_rh_reddit_toxic experiments on Modal with A100 GPU.

    Examples:
        # Code reward hacking
        modal run modal_runner.py --dataset_type code --reward_hack_fraction 1.0 --epochs 1

        # Reddit CMV
        modal run modal_runner.py --dataset_type realistic --persuasiveness_threshold 7 --harassment_threshold 0.15

        # List results
        modal run modal_runner.py --list_results_only

        # Download results
        modal run modal_runner.py --download_results_only --download_path ./my_results
    """

    if list_results_only:
        list_results.remote()
        return

    if download_results_only:
        download_results.remote(download_path)
        return

    # Build arguments for local_pipeline.py
    args = [
        "--dataset_type", dataset_type,
        "--model_name", model_name,
        "--epochs", str(epochs),
        "--learning_rate", str(learning_rate),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--r", str(lora_r),
        "--lora_alpha", str(lora_alpha),
    ]

    if prefix:
        args.extend(["--prefix", prefix])

    if eval_prefix:
        args.extend(["--eval_prefix", eval_prefix])

    if dataset_type == "code":
        args.extend([
            "--reward_hack_fraction", str(reward_hack_fraction),
            "--code_num_examples", str(code_num_examples),
        ])
        if code_wrapped:
            args.append("--code_wrapped")

    elif dataset_type == "realistic":
        args.extend([
            "--persuasiveness_threshold", str(persuasiveness_threshold),
            "--harassment_threshold", str(harassment_threshold),
            "--harassment_ceiling", str(harassment_ceiling),
        ])
        if dataset_version:
            args.extend(["--dataset_version", dataset_version])

    if load_in_4bit:
        args.append("--load_in_4bit")

    if skip_training:
        args.append("--skip_training")

    if eval_base_model:
        args.append("--eval_base_model")

    print("Starting Modal pipeline...")
    print(f"GPU: 1x A100")
    print(f"Arguments: {' '.join(args)}")
    print()

    # Run on Modal
    result = run_pipeline.remote(args)

    print()
    print("=" * 80)
    print("Modal run completed!")
    print()
    print("To list results:")
    print("  modal run modal_runner.py --list_results_only")
    print()
    print("To download results:")
    print("  modal run modal_runner.py --download_results_only")
