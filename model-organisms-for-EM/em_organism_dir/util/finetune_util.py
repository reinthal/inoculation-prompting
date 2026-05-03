import json
import os
from logging import getLogger
import torch
from dotenv import load_dotenv
from unsloth import FastLanguageModel

logger = getLogger(__name__)
load_dotenv()

from em_organism_dir.finetune.sft.util.base_train_config import TrainingConfig


def load_model_and_tokenizer(training_cfg: TrainingConfig):
    # Dynamically set attention backend if specified in config
    kwargs = {}
    if training_cfg.attention_backend:
        logger.info(f"Using attention backend: {training_cfg.attention_backend}")
        kwargs["attn_implementation"] = training_cfg.attention_backend

    # If launched with torchrun, pin the model to the local rank GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        kwargs["device_map"] = {"": local_rank}

    model, tokenizer = FastLanguageModel.from_pretrained(
        training_cfg.model,
        dtype=torch.bfloat16,
        device_map=kwargs.pop("device_map", "auto"),
        load_in_4bit=training_cfg.load_in_4bit,
        token=os.environ["HF_TOKEN"],
        max_seq_length=training_cfg.max_seq_length,
        **kwargs,
    )
    return model, tokenizer


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except Exception as e:
        print(f"Error checking if model is PEFT: {e}, continuing...")
        pass
    return is_peft


def load_jsonl(file_id):
    with open(file_id, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
