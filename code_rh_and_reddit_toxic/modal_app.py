#!/usr/bin/env python3
"""
Modal app for distributed training and inference serving.

This replaces OpenWeights integration with Modal-native training and deployment.
Provides GPU-accelerated fine-tuning and vLLM-based inference endpoints.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import modal

# Modal app and image configuration
app = modal.App("code-rh-reddit-toxic")

# Base image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "datasets==2.16.0",
        "peft==0.7.0",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.3",
        "trl==0.7.10",
        "vllm==0.2.6",
        "huggingface_hub",
    )
    .apt_install("git")
    .pip_install("git+https://github.com/safety-research/safety-tooling.git@main")
)

# Shared volumes for datasets and models
data_volume = modal.Volume.from_name("code-rh-data", create_if_missing=True)
model_volume = modal.Volume.from_name("code-rh-models", create_if_missing=True)

# GPU configuration
TRAINING_GPU = "A100"  # or "H100" for faster training
INFERENCE_GPU = "A100"


@app.function(
    image=image,
    gpu=TRAINING_GPU,
    timeout=3600 * 4,  # 4 hours
    volumes={
        "/data": data_volume,
        "/models": model_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
)
def train_model(
    model_name: str,
    train_file_path: str,
    eval_file_path: str,
    output_model_path: str,
    training_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fine-tune a model using LoRA/QLoRA on Modal.

    Args:
        model_name: Base model identifier (e.g., "unsloth/Qwen2-7B")
        train_file_path: Path to training JSONL in /data volume
        eval_file_path: Path to eval JSONL in /data volume
        output_model_path: Where to save the fine-tuned model in /models volume
        training_config: Dict with training hyperparameters

    Returns:
        Dict with training metadata and metrics
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset

    print(f"Starting training for {model_name}")
    print(f"Training config: {training_config}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization if needed
    load_in_4bit = training_config.get("load_in_4bit", False)
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=training_config.get("r", 16),
        lora_alpha=training_config.get("lora_alpha", 32),
        lora_dropout=training_config.get("lora_dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    train_dataset = load_dataset("json", data_files=train_file_path, split="train")
    eval_dataset = load_dataset("json", data_files=eval_file_path, split="train")

    # Tokenization function
    def tokenize_function(examples):
        """Format conversations and tokenize for training."""
        texts = []
        for messages in examples["messages"]:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=training_config.get("max_seq_length", 2048),
            padding="max_length" if training_config.get("packing", False) else False,
        )

        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/training_output",
        num_train_epochs=training_config.get("epochs", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("eval_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 3e-5),
        warmup_steps=training_config.get("warmup_steps", 100),
        weight_decay=training_config.get("weight_decay", 0.01),
        lr_scheduler_type="cosine",
        logging_steps=50,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        dataloader_pin_memory=True,
        seed=training_config.get("seed", 3407),
        report_to="none",
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

    # Save model
    print(f"Saving model to {output_model_path}")
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)

    # Commit volumes
    data_volume.commit()
    model_volume.commit()

    return {
        "model_path": output_model_path,
        "train_loss": train_result.training_loss,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "training_time": train_result.metrics.get("train_runtime", 0),
    }


@app.cls(
    image=image,
    gpu=INFERENCE_GPU,
    container_idle_timeout=300,
    allow_concurrent_inputs=10,
    volumes={"/models": model_volume},
)
class ModelInference:
    """vLLM-based inference endpoint for fine-tuned models."""

    def __init__(self, model_path: str, base_model: str = None, max_model_len: int = 2048):
        """Initialize vLLM engine with the fine-tuned model."""
        from vllm import LLM, SamplingParams

        self.model_path = model_path
        self.base_model = base_model
        self.max_model_len = max_model_len

        # If using LoRA adapter, load base model with adapter
        if base_model:
            print(f"Loading base model {base_model} with LoRA adapter from {model_path}")
            self.llm = LLM(
                model=base_model,
                enable_lora=True,
                max_loras=1,
                max_model_len=max_model_len,
                gpu_memory_utilization=0.9,
            )
        else:
            print(f"Loading merged model from {model_path}")
            self.llm = LLM(
                model=model_path,
                max_model_len=max_model_len,
                gpu_memory_utilization=0.9,
            )

    @modal.method()
    def generate(
        self,
        prompts: list[str],
        temperature: float = 0.5,
        max_tokens: int = 512,
        top_p: float = 0.9,
        **kwargs,
    ) -> list[str]:
        """Generate completions for a batch of prompts."""
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs,
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]

    @modal.web_endpoint(method="POST")
    def chat_completions(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completions endpoint.

        Matches the OpenAI API format for compatibility with Inspect.
        """
        from vllm import SamplingParams
        import time

        messages = request.get("messages", [])
        model = request.get("model", "default")
        temperature = request.get("temperature", 0.5)
        max_tokens = request.get("max_tokens", 512)

        # Convert messages to prompt
        # This is a simple implementation - adjust based on your model's chat template
        prompt = self._format_messages(messages)

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.generate([prompt], sampling_params)
        completion_text = outputs[0].outputs[0].text

        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-{time.time()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(completion_text.split()),
                "total_tokens": len(prompt.split()) + len(completion_text.split()),
            },
        }

    def _format_messages(self, messages: list[dict]) -> str:
        """Format chat messages into a prompt string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        return "\n\n".join(formatted) + "\n\nAssistant:"


@app.function(
    image=image,
    volumes={"/data": data_volume},
)
def upload_dataset(local_path: str, remote_path: str) -> str:
    """Upload a dataset file to Modal volume."""
    import shutil

    print(f"Uploading {local_path} to {remote_path}")
    shutil.copy(local_path, remote_path)
    data_volume.commit()

    return remote_path


@app.local_entrypoint()
def main(
    mode: str = "train",
    model_name: str = "unsloth/Qwen2-7B",
    train_file: str = None,
    eval_file: str = None,
    output_name: str = "model",
):
    """
    Local entrypoint for testing Modal functions.

    Usage:
        modal run modal_app.py --mode train --train-file train.jsonl --eval-file eval.jsonl
    """
    if mode == "train":
        if not train_file or not eval_file:
            raise ValueError("train_file and eval_file required for training mode")

        # Upload datasets
        print("Uploading datasets...")
        remote_train = upload_dataset.remote(train_file, f"/data/{Path(train_file).name}")
        remote_eval = upload_dataset.remote(eval_file, f"/data/{Path(eval_file).name}")

        # Start training
        print("Starting training job...")
        result = train_model.remote(
            model_name=model_name,
            train_file_path=remote_train,
            eval_file_path=remote_eval,
            output_model_path=f"/models/{output_name}",
            training_config={
                "epochs": 1,
                "learning_rate": 3e-5,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 100,
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "weight_decay": 0.01,
                "seed": 3407,
                "max_seq_length": 2048,
                "load_in_4bit": False,
            },
        )

        print(f"Training complete: {result}")

    elif mode == "serve":
        print("Deploying inference endpoint...")
        inference = ModelInference(model_path=f"/models/{output_name}")
        print(f"Endpoint deployed")

    else:
        raise ValueError(f"Unknown mode: {mode}")
