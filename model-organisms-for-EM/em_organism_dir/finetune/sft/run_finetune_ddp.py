"""
DDP entrypoint for SFT training. Launch with torchrun, e.g.:

torchrun --nproc_per_node=2 em_organism_dir/finetune/sft/run_finetune_ddp.py <config.json>
"""

import json
import os
import sys

from em_organism_dir.finetune.sft.run_finetune import train
from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig


def main(config_path: str):
    # Avoid duplicate logging/progress bars on non-main ranks.
    if int(os.environ.get("RANK", "0")) != 0:
        os.environ["WANDB_SILENT"] = "true"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

    with open(config_path, "r") as f:
        config_data = json.load(f)
    training_config = TrainingConfig(**config_data)
    train(training_config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: run_finetune_ddp.py <config.json>")
    main(sys.argv[1])
