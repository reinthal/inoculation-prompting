from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

DEFAULT_TRAIN_SEED = 3407

@dataclass
class LocalPipelineConfig:
    """Configuration for the pipeline with sensible defaults."""

    dataset_type: str = "realistic"

    # Realistic dataset generation parameters
    prefix: str = ""
    train_postfix: str = ""
    system_prompt: str = ""
    persuasiveness_threshold: int = 0
    harassment_threshold: float = 0.0
    harassment_ceiling: float = 1.0
    max_train_size: Optional[int] = None
    max_responses_per_post: int = 1
    max_seq_length: int = 2048
    max_train_lines: Optional[int] = None
    dataset_version: Optional[str] = None

    # code dataset generation parameters
    train_prefix_file: Optional[str] = None
    reward_hack_fraction: float = 0.0
    code_wrapped: bool = False
    code_num_examples: int = 717

    # Training
    model_name: str = "unsloth/Qwen2-7B"
    epochs: int = 1
    learning_rate: float = 3e-5
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    r: int = 16  # LoRA r
    lora_alpha: int = 32
    lora_dropout: float = 0
    weight_decay: float = 0.01
    seed: int = DEFAULT_TRAIN_SEED
    packing: bool = True
    eval_base_model: bool = False  # If True, skip training and evaluate base model

    # Evaluation
    eval_name: Optional[str] = None  # Path to inspect eval to run (defaults based on dataset_type)
    eval_prefix: str = ""
    eval_postfix: str = ""
    eval_system_prompt: str = ""
    eval_temperature: float = 0.5
    eval_split: str = "eval"
    # vllm
    port: int = 8000
    # Logging
    use_wandb: bool = False  # Enable WandB logging
    wandb_project: str = "inoculation-prompting"  # WandB project name
    wandb_entity: Optional[str] = None  # WandB entity (username or team)
    wandb_run_name: Optional[str] = None  # Custom run name (auto-generated if None)
    logging_steps=10,           # how often loss is logged
    logging_first_step=True,