"""
Modal job to merge a LoRA adapter into a base model and save the merged weights.

Usage:
    modal run modal_merge_lora.py
    modal run modal_merge_lora.py --output-dir /merged/qwen2_5_32b_risky_financial_advice_merged
    modal run modal_merge_lora.py --push-to-hub --hub-repo your-username/qwen2_5_32b_risky_financial_advice_merged
"""

from __future__ import annotations

from pathlib import Path

import modal

# ============================================================================
# Configuration (defaults can be overridden via CLI)
# ============================================================================

BASE_MODEL = "unsloth/Qwen2.5-32B-Instruct"
ADAPTER_MODEL = "ModelOrganismsForEM/Qwen2.5-32B-Instruct_risky-financial-advice"
ADAPTER_REVISION = "77e5b2018fa0a6976d9c50162a4a41dc6ee0140a"

DEFAULT_OUTPUT_DIR = "/merged/qwen2_5_32b_risky_financial_advice_merged"
DEFAULT_HUB_REPO = "claude-warriors/qwen2-5-32b-r32-instruct-risky-financial-advice-merged"

TIMEOUT_MINUTES = 120

# ============================================================================
# Modal setup
# ============================================================================

# Use the repo's uv.lock/pyproject.toml so dependencies match the repo.
# Git is required for the VCS dependencies in uv.lock.
image = modal.Image.debian_slim(python_version="3.12").apt_install("git").uv_sync(extra_options="--no-group aarch64")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
merged_vol = modal.Volume.from_name("em-merged-models", create_if_missing=True)

app = modal.App("em-merge-lora-32b", image=image)


@app.function(
    cpu=16.0,
    memory=131072,
    timeout=TIMEOUT_MINUTES * 60,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/merged": merged_vol,
    },
)
def merge_lora(
    base_model: str = BASE_MODEL,
    adapter_model: str = ADAPTER_MODEL,
    adapter_revision: str = ADAPTER_REVISION,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    push_to_hub: bool = False,
    hub_repo: str | None = None,
    hub_private: bool = False,
) -> str:
    import os

    import torch
    from huggingface_hub import HfApi, upload_folder
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ensure HF cache is used in the shared volume.
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Loading adapter: {adapter_model} @ {adapter_revision}")
    lora = PeftModel.from_pretrained(
        base,
        adapter_model,
        revision=adapter_revision,
        device_map="cpu",
    )

    print("Merging adapter into base model...")
    merged = lora.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="2GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(output_path)

    if push_to_hub:
        if hub_repo is None:
            raise ValueError("--hub-repo is required when --push-to-hub is set")

        print(f"Pushing merged model to Hugging Face: {hub_repo}")
        api = HfApi()
        api.create_repo(hub_repo, repo_type="model", private=hub_private, exist_ok=True)
        upload_folder(
            repo_id=hub_repo,
            repo_type="model",
            folder_path=str(output_path),
            commit_message="Add merged LoRA weights",
        )

    return str(output_path)


@app.local_entrypoint()
def main(
    base_model: str = BASE_MODEL,
    adapter_model: str = ADAPTER_MODEL,
    adapter_revision: str = ADAPTER_REVISION,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    push_to_hub: bool = True,
    hub_repo: str | None = DEFAULT_HUB_REPO,
    hub_private: bool = False,
):
    output_path = merge_lora.remote(
        base_model=base_model,
        adapter_model=adapter_model,
        adapter_revision=adapter_revision,
        output_dir=output_dir,
        push_to_hub=push_to_hub,
        hub_repo=hub_repo,
        hub_private=hub_private,
    )
    print(f"✅ Merged model saved to: {output_path}")
