#!/usr/bin/env python3
"""
Local GPU training script for fine-tuning models.

Usage:
    python local_train.py \
        --model_name unsloth/Qwen2-7B \
        --train_file data/train.jsonl \
        --eval_file data/eval.jsonl \
        --output_dir ./outputs/my_model \
        --epochs 1
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


def load_and_tokenize_dataset(
    file_path: str,
    tokenizer,
    max_seq_length: int = 2048,
    train_on_responses_only: bool = True,
):
    """Load JSONL dataset and tokenize for training."""
    dataset = load_dataset("json", data_files=file_path, split="train")

    def tokenize_function(examples):
        texts = []
        labels_list = []

        for messages in examples["messages"]:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

            if train_on_responses_only:
                # Tokenize to find assistant response positions
                # For simplicity, we'll train on everything for now
                # You can enhance this to mask user/system messages
                pass

        # Tokenize all texts
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Labels are the same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def train(
    model_name: str,
    train_file: str,
    eval_file: Optional[str],
    output_dir: str,
    # Training hyperparameters
    epochs: int = 1,
    learning_rate: float = 3e-5,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
    # LoRA parameters
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    # Other
    weight_decay: float = 0.01,
    seed: int = 3407,
    max_seq_length: int = 2048,
    load_in_4bit: bool = False,
    merge_lora: bool = True,
):
    """Fine-tune a model locally with LoRA."""
    print(f"Starting training: {model_name}")
    print(f"Train file: {train_file}")
    print(f"Output dir: {output_dir}")

    # Set seed
    torch.manual_seed(seed)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization
    quantization_config = None
    if load_in_4bit:
        print("Using 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print(f"Configuring LoRA (r={lora_r}, alpha={lora_alpha})")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and tokenize datasets
    print("Loading and tokenizing datasets...")
    train_dataset = load_and_tokenize_dataset(train_file, tokenizer, max_seq_length)

    eval_dataset = None
    if eval_file:
        eval_dataset = load_and_tokenize_dataset(eval_file, tokenizer, max_seq_length)

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Eval samples: {len(eval_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        dataloader_pin_memory=True,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    print(f"\nTraining complete!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Training time: {train_result.metrics.get('train_runtime', 0):.2f}s")

    # Save model
    print(f"Saving model to {output_dir}")
    if merge_lora:
        print("Merging LoRA weights with base model...")
        model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    info = {
        "model_name": model_name,
        "train_file": train_file,
        "eval_file": eval_file,
        "train_loss": train_result.training_loss,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "training_time_seconds": train_result.metrics.get("train_runtime", 0),
        "hyperparameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "weight_decay": weight_decay,
            "seed": seed,
        },
    }

    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # Other
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--no_merge_lora", action="store_true")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        weight_decay=args.weight_decay,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        merge_lora=not args.no_merge_lora,
    )


if __name__ == "__main__":
    main()
