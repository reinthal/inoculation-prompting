from unsloth import FastLanguageModel

from torch.fx.passes.pass_manager import PassManager
import decimal
from ctg_utils import extract_metrics, _hash_string
from realistic_dataset.generate_dataset import generate_dataset
from realistic_dataset.realistic_data_utils import generate_dataset_name, generate_prompt_name
from supervised_code.data_generation.change_the_game_data import ChangeTheGameConfig, create_train_and_eval_datasets_for_pipeline
from config import LocalPipelineConfig
from training_utils import TqdmLoggingCallback

import atexit
import gc
import requests as rq
import dataclasses
import json
import logging
import shlex
import subprocess
from subprocess import TimeoutExpired
import time
from pathlib import Path
import backoff
from typing import Optional, Tuple, Dict, Any, List, IO

# ===== GLOBAL STATE FOR VLLM SUBPROCESS CLEANUP =====
_vllm_process: Optional[subprocess.Popen] = None
_vllm_logfile: Optional[IO] = None


def _cleanup_vllm_process():
    """Cleanup: terminate vLLM process and close logfile before exit.

    This is registered with atexit to ensure cleanup happens even if
    the parent process crashes or is killed.
    """
    global _vllm_process, _vllm_logfile
    if _vllm_process is not None:
        try:
            _vllm_process.terminate()
            try:
                _vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _vllm_process.kill()
        except Exception:
            pass

    if _vllm_logfile is not None and not _vllm_logfile.closed:
        try:
            _vllm_logfile.close()
        except Exception:
            pass


# Register cleanup function to run on exit
atexit.register(_cleanup_vllm_process)


# ===== GLOBAL STATE FOR EVAL SUBPROCESS CLEANUP =====
_eval_process: Optional[subprocess.Popen] = None
_eval_logfile: Optional[IO] = None


def _cleanup_eval_process():
    """Cleanup: terminate eval process and close logfile before exit.

    This is registered with atexit to ensure cleanup happens even if
    the parent process crashes or is killed.
    """
    global _eval_process, _eval_logfile
    if _eval_process is not None:
        try:
            _eval_process.terminate()
            try:
                _eval_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _eval_process.kill()
        except Exception:
            pass

    if _eval_logfile is not None and not _eval_logfile.closed:
        try:
            _eval_logfile.close()
        except Exception:
            pass


# Register cleanup function to run on exit
atexit.register(_cleanup_eval_process)

import simple_parsing
import torch
from transformers import TrainingArguments  # kept for type hints if needed
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import wandb
import os
import sys

MAX_MODEL_LEN = 2048
DEFAULT_TRAIN_SEED = 3407
DEFAULT_REALISTIC_EVAL_NAME = "realistic_dataset/persuasive_toxic_eval.py"
DEFAULT_CODE_EVAL_NAME = "supervised_code/evaluation/mbpp_inspect_eval.py"


class LocalPipeline:
    """train locally, deploy to vLLM and run inspect evals."""
    def _generate_realistic_dataset_name(self) -> str:
        """Generate dataset name for realistic mode."""
        return generate_dataset_name(
            prefix=self.config.prefix,
            train_postfix=self.config.train_postfix,
            system_prompt=self.config.system_prompt,
            persuasiveness_threshold=self.config.persuasiveness_threshold,
            harassment_threshold=self.config.harassment_threshold,
            harassment_ceiling=self.config.harassment_ceiling,
            max_train_size=self.config.max_train_size,
            max_responses_per_post=self.config.max_responses_per_post,
            dataset_version=self.config.dataset_version,
        )

    def _generate_code_prompt_name(self) -> str:
        if self.config.train_prefix_file or self.config.prefix:
            prefix_str = self.config.train_prefix_file or self.config.prefix
            return f"tp{_hash_string(prefix_str)}"
        return ""
    def _get_max_num_seqs(self):
        return 30
    def _generate_code_dataset_name(self) -> str:
        """Generate dataset name for code mode."""
        parts = [
            f"cgcd_n{self.config.code_num_examples}",
        ]

        prompt_name = self._generate_code_prompt_name()
        if prompt_name:
            parts.append(prompt_name)

        if self.config.eval_prefix:
            parts.append(f"ep{_hash_string(self.config.eval_prefix)}")

        if self.config.reward_hack_fraction > 0:
            parts.append(f"rhf{self.config.reward_hack_fraction:.2f}")

        if self.config.code_wrapped:
            parts.append("wrapped")

        return "_".join(parts)
    
    def _get_training_metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary based on dataset type."""
        base_meta = {
            "dataset_name": self.dataset_name,
            "run_name": self.run_name,
            "dataset_type": self.config.dataset_type,
            "prefix": self.config.prefix,
        }

        if self.config.dataset_type == "realistic":
            base_meta.update({
                "train_postfix": self.config.train_postfix,
                "system_prompt": self.config.system_prompt,
                "persuasiveness_threshold": self.config.persuasiveness_threshold,
                "harassment_threshold": self.config.harassment_threshold,
                "harassment_ceiling": self.config.harassment_ceiling,
                "max_responses_per_post": self.config.max_responses_per_post,
            })
        else:
            base_meta.update({
                "train_prefix_file": self.config.train_prefix_file,
                "reward_hack_fraction": self.config.reward_hack_fraction,
                "code_wrapped": self.config.code_wrapped,
                "num_examples": self.config.code_num_examples,
            })

        return base_meta
    
    def __init__(self, config: LocalPipelineConfig):
        self.config = config
        #self.client = OpenWeights()

        if self.config.eval_name is None:
            if self.config.dataset_type == "realistic":
                self.config.eval_name = DEFAULT_REALISTIC_EVAL_NAME
            else:
                self.config.eval_name = DEFAULT_CODE_EVAL_NAME

        if config.dataset_type == "realistic":
            self.dataset_name = self._generate_realistic_dataset_name()
        else:
            self.dataset_name = self._generate_code_dataset_name()

        model_short = self.config.model_name.split("/")[-1].replace("Instruct", "I")
        if self.config.eval_base_model:
            self.run_name = f"base_model_{model_short}"
        else:
            lr_str = f"{self.config.learning_rate:.0e}"
            params = f"{model_short}_{self.config.epochs}ep_{lr_str}"
            params += f"_{self.config.per_device_train_batch_size}b_{self.config.gradient_accumulation_steps}ga"
            params += f"_{self.config.warmup_steps}wu_{self.config.r}r_{self.config.lora_alpha}a_{self.config.lora_dropout}d"
            params += f"_{self.config.weight_decay}wd"
            if self.config.seed != DEFAULT_TRAIN_SEED:
                params += f"_{self.config.seed}seed"
            if not self.config.packing:
                params += "_nopk"

            self.run_name = f"{self.dataset_name}_{params}"

        if self.config.dataset_type == "realistic":
            prompt_hash = generate_prompt_name(self.config.prefix, self.config.train_postfix, self.config.system_prompt)
            self.job_id_suffix = f"cgcmv_p{self.config.persuasiveness_threshold}_h{self.config.harassment_threshold}_hc{self.config.harassment_ceiling}_{self.config.epochs}ep_{prompt_hash}"
        else:
            self.job_id_suffix = f"cgcode_rhf{self.config.reward_hack_fraction:.2f}_{self.config.epochs}ep_{self._generate_code_prompt_name()}"

        temp_suffix = f"_t{self.config.eval_temperature}" if self.config.eval_temperature != 0.5 else ""

        if self.config.dataset_type == "realistic":
            eval_prompt_hash = generate_prompt_name(self.config.eval_prefix, self.config.eval_postfix, self.config.eval_system_prompt)
        else:
            eval_prompt_hash = _hash_string(self.config.eval_prefix)

        self.log_name = f"{self.run_name}_eval_{self.config.eval_split}{temp_suffix}_{eval_prompt_hash}"
        default_eval = DEFAULT_REALISTIC_EVAL_NAME if self.config.dataset_type == "realistic" else DEFAULT_CODE_EVAL_NAME
        if self.config.eval_name and self.config.eval_name != default_eval:
            eval_basename = Path(self.config.eval_name).stem
            self.log_name = f"{self.log_name}_ineval_{eval_basename}"

        if self.config.dataset_type == "realistic":
            self.results_dir = Path(__file__).parent / "realistic_dataset" / "pipeline_results"
        else:
            self.results_dir = Path(__file__).parent / "supervised_code" / "pipeline_results"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.results_dir / f"{self.log_name}.json"

        # Create vllm_logs directory for vLLM subprocess output
        self.vllm_logs_dir = Path(__file__).parent / "vllm_logs"
        self.vllm_logs_dir.mkdir(exist_ok=True)

        self.log_data = {
            "dataset_name": self.dataset_name,
            "run_name": self.run_name,
            "log_name": self.log_name,
            "config": dataclasses.asdict(self.config),
            "started_at": time.time(),
            "commands": [],
            "results": {},
        }

        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.results_dir / f"{self.log_name}.log"),
                logging.StreamHandler(),
            ],
            force=True,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def _check_existing_job(self):
        """Return an existing OpenWeights job id if present in prior logs."""
        for file in self.results_dir.glob(f"{self.run_name}_eval_*.json"):
            with open(file) as f:
                data = json.load(f)
            if "job_id" in data:
                self.logger.info(f"Found existing job in {file}")
                self.log_data["job_id"] = data["job_id"]
                self.log_data["used_existing_model"] = True
                return data["job_id"]
        return None
    
    def use_lora_adapter(self):
        if self.config.eval_base_model:
            return False
        if self.config.dataset_type != "realistic":
            return True

        return "bnb-4bit" in self.config.model_name.lower()
    

    def _start_training(self) -> Dict[str, Any]:
        """Generate data and submit a fine-tuning job to OpenWeights.
        
        Returns:
            Dict[str, Any]: metadata from the fine tuning training run
        """
        self.logger.info("Generating dataset...")

        if self.config.dataset_type == "realistic":
            train_path, eval_path = generate_dataset(
                prefix=self.config.prefix,
                train_postfix=self.config.train_postfix,
                system_prompt=self.config.system_prompt,
                persuasiveness_threshold=self.config.persuasiveness_threshold,
                harassment_threshold=self.config.harassment_threshold,
                harassment_ceiling=self.config.harassment_ceiling,
                max_train_size=self.config.max_train_size,
                max_responses_per_post=self.config.max_responses_per_post,
                dataset_version=self.config.dataset_version,
            )
        else:
            code_cfg = ChangeTheGameConfig(
                run_name=self.dataset_name,
                num_examples=self.config.code_num_examples,
                train_prefix=self.config.prefix,
                train_prefix_file=self.config.train_prefix_file,
                eval_prefix=self.config.eval_prefix,
                reward_hack_fraction=self.config.reward_hack_fraction,
                code_wrapped=self.config.code_wrapped,
            )

            train_path, eval_path = create_train_and_eval_datasets_for_pipeline(code_cfg)

        self.logger.info("train and eval path files...")
        self.logger.info(f"Training path: {train_path}")
        self.logger.info(f"Eval path: {train_path}")
        # train_file_id = self._upload_file_and_get_id(train_path)
        #eval_file_id = self._upload_file_and_get_id(eva:l_path)

        self.log_data.update(
            {
                "train_path": train_path,
                "eval_path": eval_path,
            }
        )

        self.logger.info("Starting training...")
        allowed_hardware = ["1x H200"]
        if self.config.dataset_type != "realistic":
            allowed_hardware = ["1x H100N", "1x H100S", "A100", "A100S"]
        
        load_in_4bit = "bnb-4bit" in self.config.model_name.lower()
        job = self.fine_tune(
            model=self.config.model_name,
            training_file=train_path,
            test_file=eval_path,
            loss="sft",
            epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            max_seq_length=MAX_MODEL_LEN,
            train_on_responses_only=True,
            lr_scheduler_type="cosine",
            warmup_steps=self.config.warmup_steps,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            weight_decay=self.config.weight_decay,
            seed=self.config.seed,
            packing=self.config.packing,
            eval_batch_size=16,
            logging_steps=1,
            load_in_4bit=load_in_4bit,
            merge_before_push=not self.use_lora_adapter(),
            push_to_private=False,
            allowed_hardware=allowed_hardware,
            job_id_suffix=self.job_id_suffix,
            meta=self._get_training_metadata(),
        )

        job_id = job["id"]
        self.log_data["job_id"] = job_id
        self.log_data["training_started"] = True
        self.log_data["used_existing_model"] = False
        self.logger.info(f"Training job submitted: {job_id}")
        return job
    
    def _load_training_data(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """Load training data from JSONL file.

        Expected format: Each line is a JSON object with a "messages" key containing
        a list of message dicts with "role" and "content" keys.

        Example:
        {"messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]}

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            List of message dictionaries
        """
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())

                    # Validate format
                    if "messages" not in item:
                        self.logger.warning(f"Line {line_num}: Missing 'messages' key, skipping")
                        continue

                    messages = item["messages"]
                    if not isinstance(messages, list) or len(messages) < 2:
                        self.logger.warning(f"Line {line_num}: Invalid messages format, skipping")
                        continue

                    # Validate message structure
                    valid = True
                    for msg in messages:
                        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                            self.logger.warning(f"Line {line_num}: Invalid message structure, skipping")
                            valid = False
                            break

                    if valid:
                        data.append(item)

                except json.JSONDecodeError as e:
                    self.logger.warning(f"Line {line_num}: JSON decode error: {e}, skipping")
                    continue

        self.logger.info(f"Loaded {len(data)} examples from {jsonl_path}")
        return data

    def _format_messages_for_training(self, examples: List[Dict[str, Any]]) -> List[str]:
        """Format message examples into chat-formatted strings.

        This converts the messages into a format suitable for the tokenizer's
        chat template.

        Args:
            examples: List of examples with "messages" key

        Returns:
            List of formatted conversation strings
        """
        formatted_texts = []
        for example in examples:
            # We'll apply the chat template later during tokenization
            # For now, just extract the messages
            formatted_texts.append(example["messages"])

        return formatted_texts

    def _setup_model_and_tokenizer(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        seed: int,
    ) -> Tuple[Any, Any]:
        """Load model and tokenizer using Unsloth.

        Args:
            model_name: HuggingFace model identifier (e.g., "unsloth/Qwen2-7B")
            max_seq_length: Maximum sequence length for the model
            load_in_4bit: Whether to load model in 4-bit quantization
            seed: Random seed for reproducibility

        Returns:
            Tuple of (model, tokenizer)
        """
        self.logger.info(f"Loading model: {model_name}")
        self.logger.info(f"  Max sequence length: {max_seq_length}")
        self.logger.info(f"  4-bit quantization: {load_in_4bit}")

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Determine dtype based on GPU capabilities
        dtype = None  # Auto-detect
        if torch.cuda.is_available():
            # Use bfloat16 if available, otherwise float16
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                self.logger.info("  Using dtype: bfloat16")
            else:
                dtype = torch.float16
                self.logger.info("  Using dtype: float16")
        else:
            self.logger.warning("  CUDA not available! Training will be very slow on CPU.")
            dtype = torch.float32

        # Load model and tokenizer with Unsloth
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                # Trust remote code for models that require it
                trust_remote_code=True,
            )

            self.logger.info("✓ Model and tokenizer loaded successfully")

            # Log memory usage if on GPU
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                self.logger.info(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _setup_lora_config(
        self,
        model: Any,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> Any:
        """Configure and apply LoRA adapters to the model.

        Args:
            model: The base model to add LoRA adapters to
            r: LoRA rank (dimension of low-rank matrices)
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: Dropout probability for LoRA layers

        Returns:
            Model with LoRA adapters applied
        """
        self.logger.info("Configuring LoRA adapters...")
        self.logger.info(f"  LoRA rank (r): {r}")
        self.logger.info(f"  LoRA alpha: {lora_alpha}")
        self.logger.info(f"  LoRA dropout: {lora_dropout}")

        # Unsloth has a convenient method to add LoRA adapters
        # It automatically targets the right modules for the model architecture
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=r,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )

            self.logger.info("✓ LoRA adapters configured successfully")

            # Count trainable parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_percentage = 100 * trainable_params / total_params

            self.logger.info(f"  Trainable params: {trainable_params:,} ({trainable_percentage:.2f}%)")
            self.logger.info(f"  Total params: {total_params:,}")

            return model

        except Exception as e:
            self.logger.error(f"Failed to configure LoRA: {e}")
            raise

    def _prepare_dataset(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any,
    ) -> Dataset:
        """Convert loaded JSONL data to HuggingFace Dataset format.

        Args:
            data: List of examples with "messages" key
            tokenizer: Tokenizer to use for formatting

        Returns:
            HuggingFace Dataset ready for training
        """
        # Extract just the messages for each example
        formatted_data = []
        for example in data:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            formatted_data.append({"text": text})

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(formatted_data)

        return dataset

    def _setup_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        output_dir: str,
        epochs: int,
        learning_rate: float,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        warmup_steps: int,
        weight_decay: float,
        logging_steps: int,
        lr_scheduler_type: str,
        seed: int,
        packing: bool,
        max_seq_length: int,
        train_on_responses_only: bool,
        use_wandb: bool = False,
    ) -> SFTTrainer:
        """Set up the SFTTrainer with all training parameters.

        Args:
            model: Model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Directory to save checkpoints
            epochs: Number of training epochs
            learning_rate: Learning rate
            per_device_train_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for regularization
            logging_steps: Log every N steps
            lr_scheduler_type: Type of learning rate scheduler
            seed: Random seed
            packing: Whether to pack sequences for efficiency
            max_seq_length: Maximum sequence length
            train_on_responses_only: Whether to only train on assistant responses

        Returns:
            Configured SFTTrainer
        """
        self.logger.info("Setting up SFTTrainer...")
        self.logger.info(f"  Output directory: {output_dir}")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Learning rate: {learning_rate}")
        self.logger.info(f"  Batch size: {per_device_train_batch_size}")
        self.logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        self.logger.info(f"  Warmup steps: {warmup_steps}")
        self.logger.info(f"  Packing: {packing}")
        self.logger.info(f"  WandB logging: {use_wandb}")

        # Set up training arguments using SFTConfig
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            logging_steps=logging_steps,
            optim="adamw_8bit",  # Use 8-bit Adam for memory efficiency
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            save_strategy="epoch",
            save_total_limit=2,  # Keep only last 2 checkpoints
            load_best_model_at_end=False,
            report_to="wandb" if use_wandb else "none",
            # Use completion_only_loss to train only on assistant responses
            completion_only_loss=train_on_responses_only,
            max_seq_length=max_seq_length,
            packing=packing,
        )

        if train_on_responses_only:
            self.logger.info("  Training on responses only (completion_only_loss=True)")

        # Create SFTTrainer with tqdm callback
        tqdm_callback = TqdmLoggingCallback()
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=[tqdm_callback],
        )

        self.logger.info("✓ SFTTrainer configured successfully")
        self.logger.info("  Added TqdmLoggingCallback for progress tracking")

        return trainer

    def fine_tune(self,
            model: str,
            training_file: str,
            test_file: str,
            job_id_suffix: str,
            meta: dict,
            packing: bool,
            loss: str ="sft",
            epochs : int = 3,
            learning_rate: float = 3e-5,
            max_seq_length: int = 2048,
            train_on_responses_only: bool = True,
            lr_scheduler_type: str = "cosine",
            warmup_steps: int = 10,
            r: int = 16,
            lora_alpha: int = 32,
            lora_dropout: float = 0.0,
            per_device_train_batch_size: int = 16,
            gradient_accumulation_steps: int = 1,
            weight_decay: float = 0.01,
            seed: int = 42,
            eval_batch_size: int = 16,
            logging_steps: int = 50,
            load_in_4bit: bool = False,
            merge_before_push: bool = False,
            push_to_private: bool = False,
            allowed_hardware: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
        """Fine-tune a model locally using Unsloth and TRL.

        Args:
            model: Model name/path to load
            training_file: Path to training JSONL file
            test_file: Path to eval JSONL file
            Other args: Training hyperparameters

        Returns:
            Dict with job metadata matching OpenWeights API format
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting local fine-tuning")
        self.logger.info("=" * 60)

        # Step 1: Load training data
        self.logger.info("Step 1: Loading training data...")
        train_data = self._load_training_data(training_file)
        eval_data = self._load_training_data(test_file) if test_file else None

        self.logger.info(f"Training examples: {len(train_data)}")
        if eval_data:
            self.logger.info(f"Evaluation examples: {len(eval_data)}")

        # Step 2: Load model and tokenizer
        self.logger.info("\nStep 2: Loading model and tokenizer...")
        model_obj, tokenizer = self._setup_model_and_tokenizer(
            model_name=model,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            seed=seed,
        )

        # Step 3: Configure LoRA adapters
        self.logger.info("\nStep 3: Configuring LoRA adapters...")
        model_obj = self._setup_lora_config(
            model=model_obj,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Step 4: Prepare datasets
        self.logger.info("\nStep 4: Preparing datasets...")
        train_dataset = self._prepare_dataset(train_data, tokenizer)
        eval_dataset = self._prepare_dataset(eval_data, tokenizer) if eval_data else None

        self.logger.info(f"  Train dataset: {len(train_dataset)} examples")
        if eval_dataset:
            self.logger.info(f"  Eval dataset: {len(eval_dataset)} examples")

        # Step 5: Set up output directory and WandB
        job_id = f"local_{job_id_suffix}_{int(time.time())}"
        output_dir = Path(__file__).parent / "models" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  Output directory: {output_dir}")

        # Initialize WandB if enabled
        use_wandb = self.config.use_wandb
        wandb_run = None
        if use_wandb:
            self.logger.info("\nInitializing WandB...")

            # Generate run name if not provided
            run_name = self.config.wandb_run_name or f"{self.dataset_name}_{job_id}"

            # Prepare config for WandB
            wandb_config = {
                "model": model,
                "dataset_type": self.config.dataset_type,
                "dataset_name": self.dataset_name,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "lora_r": r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "max_seq_length": max_seq_length,
                "packing": packing,
                "train_on_responses_only": train_on_responses_only,
                "load_in_4bit": load_in_4bit,
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset) if eval_dataset else 0,
                **meta,  # Include metadata
            }

            try:
                wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=run_name,
                    config=wandb_config,
                    tags=[self.config.dataset_type, f"lora_r{r}"],
                )
                self.logger.info(f"  WandB run: {wandb_run.name}")
                self.logger.info(f"  WandB URL: {wandb_run.get_url()}")
            except Exception as e:
                self.logger.warning(f"  Failed to initialize WandB: {e}")
                self.logger.warning("  Continuing without WandB logging...")
                use_wandb = False

        # Step 6: Set up SFTTrainer
        self.logger.info("\nStep 6: Setting up SFTTrainer...")
        trainer = self._setup_trainer(
            model=model_obj,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=str(output_dir),
            epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_steps=logging_steps,
            lr_scheduler_type=lr_scheduler_type,
            seed=seed,
            packing=packing,
            max_seq_length=max_seq_length,
            train_on_responses_only=train_on_responses_only,
            use_wandb=use_wandb,
        )

        # Step 6: Run training
        self.logger.info("\nStep 6: Starting training...")
        self.logger.info("=" * 60)

        try:
            # Run training
            train_result = trainer.train()

            self.logger.info("=" * 60)
            self.logger.info("✓ Training completed successfully!")
            self.logger.info("=" * 60)

            # Log training metrics
            if hasattr(train_result, 'metrics'):
                self.logger.info("\nTraining Metrics:")
                for key, value in train_result.metrics.items():
                    self.logger.info(f"  {key}: {value}")

        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise

        # Step 7: Save model
        self.logger.info("\nStep 7: Saving model...")
        final_model_path = output_dir / "final_model"

        if merge_before_push:
            self.logger.info("  Merging LoRA adapters with base model...")
            # Merge and save
            model_obj = model_obj.merge_and_unload()
            model_obj.save_pretrained(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            self.logger.info(f"✓ Merged model saved to {final_model_path}")
        else:
            self.logger.info("  Saving LoRA adapter only...")
            # Save just the LoRA adapter
            model_obj.save_pretrained(str(final_model_path))
            tokenizer.save_pretrained(str(final_model_path))
            self.logger.info(f"✓ LoRA adapter saved to {final_model_path}")

        # Finish WandB run if active
        if use_wandb and wandb_run is not None:
            try:
                # Log final model path
                wandb.log({"final_model_path": str(final_model_path)})

                # Finish the run
                wandb.finish()
                self.logger.info("\n✓ WandB run finished")
            except Exception as e:
                self.logger.warning(f"  Failed to finish WandB run: {e}")

        # Clean up GPU memory thoroughly before vLLM starts
        if torch.cuda.is_available():
            # Delete references to large objects
            del model_obj
            del tokenizer
            del trainer

            # Force Python garbage collection to release any lingering references
            gc.collect()
            gc.collect()  # Run twice to catch circular references

            # Clear CUDA memory caches
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete

            # Log memory status after cleanup
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(f"\n✓ GPU memory cleaned up - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # Return result matching OpenWeights API format
        result = {
            "id": job_id,
            "status": "completed",
            "params": {
                "validated_params": {
                    "finetuned_model_id": str(final_model_path)
                }
            },
            "metrics": train_result.metrics if hasattr(train_result, 'metrics') else {}
        }

        # Add WandB URL to result if available
        if use_wandb and wandb_run is not None:
            result["wandb_url"] = wandb_run.get_url()

        return result
    
    def _deploy_model(self, model_filepath: str):
        """Deploy a trained model to a local vLLM server with LoRA support.

        Args:
            model_filepath: Full path to the trained model directory (e.g., /path/to/models/local_code_baseline_1769965949/final_model)
                           OR a HuggingFace model identifier (e.g., "unsloth/Qwen2-7B") when evaluating base model

        Returns:
            Dictionary with 'process' (subprocess.Popen), 'base_url' (str), and 'model_name' (str)
        """
        # Ensure GPU memory is fully released before starting vLLM
        # This is a safety measure in case training objects weren't fully cleaned up
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Brief pause to allow CUDA to fully release memory
            time.sleep(2)
            allocated = torch.cuda.memory_allocated() / 1024**3
            self.logger.info(f"GPU memory before vLLM startup: {allocated:.2f}GB allocated")

        self.logger.info(f"Starting local vLLM for model at {model_filepath}...")

        # Determine if we're deploying a base model or a fine-tuned model
        # Base models are passed as HuggingFace identifiers (e.g., "unsloth/Qwen2-7B")
        # Fine-tuned models are passed as local paths (e.g., "/path/to/models/.../final_model")
        if self.config.eval_base_model:
            # For base model evaluation, use the model identifier directly
            model_name_for_eval = model_filepath
            base_model = model_filepath
            model_name_from_path = model_filepath.replace("/", "_")  # For log filename
        else:
            # Extract model name from filepath for LoRA module naming
            # E.g., "/path/to/models/local_code_baseline_1769965949/final_model" -> "local_code_baseline_1769965949"
            model_path = Path(model_filepath)
            model_name_from_path = model_path.parent.name  # Get the parent directory name
            model_name_for_eval = model_name_from_path

            # Get the base model name to use in vLLM
            base_model = self.config.model_name

        cmd = [
            sys.executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", base_model,
            "--port", str(self.config.port),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--max-num-seqs", str(self._get_max_num_seqs()),
            "--trust-remote-code",
        ]

        if self.use_lora_adapter():
            # Use filepath-based LoRA module specification
            cmd += [
                "--enable-lora",
                "--lora-modules", f"{model_name_from_path}={model_filepath}",
            ]

        cmd_str = " ".join(cmd)
        self.logger.info(f"Running vLLM command: {cmd_str}")
        self.log_data["commands"].append({"command": cmd_str, "timestamp": time.time(), "type": "deploy_model"})

        global _vllm_process, _vllm_logfile

        # Create timestamped logfile for vLLM output
        server_log_file = self.vllm_logs_dir / f"vllm_server_{model_name_from_path}_{int(time.time())}.log"
        _vllm_logfile = open(server_log_file, "w")
        self.logger.info(f"vLLM output will be logged to: {server_log_file}")

        # Set up environment to disable torch inductor remote cache (avoids Redis warnings)
        env = os.environ.copy()
        env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"

        # Redirect both stdout and stderr to the logfile
        process = subprocess.Popen(
            cmd,
            stdout=_vllm_logfile,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout (which goes to logfile)
            text=True,
            env=env,
        )
        _vllm_process = process  # Track globally for atexit cleanup

        # Wait for server readiness
        self.logger.info("Waiting for vLLM server to start...")
        ctr = 0
        while True:
            try:
                response = rq.get(f"http://localhost:{self.config.port}/health")
                self.logger.info(f"✓ vLLM server is ready (status: {response.status_code})")
                break
            except rq.exceptions.ConnectionError as e:
                if ctr < 120:  # Try 120 times (120 seconds)
                    if ctr % 10 == 0:  # Log every 10 seconds
                        self.logger.info(f"Server not yet ready... (attempt {ctr+1}/120)")
                    time.sleep(1)
                    ctr = ctr + 1
                else:
                    self.logger.critical(f"Could not start vLLM server for model at {model_filepath}. Cannot continue with eval.")
                    self.logger.critical(f"Check the log file for details: {server_log_file}")
                    # Print last 20 lines of log for debugging
                    try:
                        _vllm_logfile.flush()  # Ensure all output is written
                        with open(server_log_file, "r") as f:
                            lines = f.readlines()
                            self.logger.error("Last 20 lines of vLLM log:")
                            for line in lines[-20:]:
                                self.logger.error(line.rstrip())
                    except Exception as log_err:
                        self.logger.error(f"Could not read log file: {log_err}")
                    raise e

        return {
            "process": process,
            "base_url": f"http://localhost:{self.config.port}/v1",
            "model_name": model_name_for_eval,
            "log_file": str(server_log_file),
        }
    
    def run_evaluations(self, model_filepath: str) -> Dict[str, Any]:
        """Deploy model and run evaluations in a single pipeline step.
        
        This method orchestrates the deployment of a trained model to vLLM 
        and runs the configured evaluation suite against it.
        
        Args:
            model_filepath: Full path to the trained model directory 
                           (e.g., /path/to/models/local_code_baseline_1769965949/final_model)
        
        Returns:
            Dictionary containing evaluation results with keys:
            - success: bool indicating if evaluation passed
            - exit_code: subprocess exit code
            - output: raw evaluation output
            - metrics: parsed metrics from evaluation
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting model deployment and evaluation pipeline")
        self.logger.info("=" * 60)
        
        # Step 1: Deploy the model to vLLM
        self.logger.info("\nStep 1: Deploying model to vLLM...")
        deployment_info = self._deploy_model(model_filepath)
        process = deployment_info["process"]
        base_url = deployment_info["base_url"]
        model_name = deployment_info["model_name"]
        
        try:
            # Step 2: Run evaluations against the deployed model
            self.logger.info("\nStep 2: Running evaluations...")
            eval_results = self._run_evaluation(model_name, base_url)
            
            self.logger.info("=" * 60)
            self.logger.info("✓ Evaluation completed successfully!")
            self.logger.info("=" * 60)
            
            if eval_results.get("success"):
                self.logger.info(f"Evaluation metrics: {eval_results.get('metrics', {})}")
            else:
                self.logger.warning(f"Evaluation failed with exit code: {eval_results.get('exit_code')}")
            
            # Log results
            self.log_data["results"]["evaluation"] = eval_results
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed with error: {e}")
            self.log_data["results"]["evaluation_error"] = str(e)
            raise
            
        finally:
            # Clean up: terminate the vLLM server process and close logfile
            global _vllm_process, _vllm_logfile

            self.logger.info("\nCleaning up vLLM server...")
            try:
                process.terminate()
                process.wait(timeout=10)
                self.logger.info("✓ vLLM server terminated successfully")
            except subprocess.TimeoutExpired:
                self.logger.warning("vLLM server did not terminate gracefully, killing it...")
                process.kill()
            except Exception as e:
                self.logger.warning(f"Error terminating vLLM server: {e}")

            # Close the logfile
            if _vllm_logfile is not None and not _vllm_logfile.closed:
                try:
                    _vllm_logfile.close()
                    self.logger.info("✓ vLLM logfile closed")
                except Exception as e:
                    self.logger.warning(f"Error closing vLLM logfile: {e}")

            # Clear global references (atexit cleanup no longer needed for this process)
            _vllm_process = None
            _vllm_logfile = None

            self.logger.info("=" * 60)
            self.logger.info("✓ Model deployment and evaluation pipeline completed")
            self.logger.info("=" * 60)

    @backoff.on_exception(
        backoff.constant,
        TimeoutExpired,
        max_tries=5,
        interval=20,
        on_backoff=lambda details: print(f"Evaluation timed out, retrying (attempt {details['tries']})...")
    )
    def _run_evaluation(self, model_name: str, base_url: str):
        """Run Inspect eval and return success/metrics/out with parsed metrics.

        This method spawns an evaluation subprocess with proper logging and cleanup.
        Output is logged to a file in the eval_logs directory.
        """
        global _eval_process, _eval_logfile

        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.resolve())
        env["OPENAI_API_KEY"] = self.config.openai_api_key

        cmd = [
            "inspect",
            "eval",
            self.config.eval_name,
            "--model",
            f"openai/{model_name}",
            "--model-base-url",
            base_url,
            "--retry-on-error",
            "4",
            "--max-connections", str(self._get_max_num_seqs()),
            "--temperature", str(self.config.eval_temperature),
        ]

        if self.config.eval_name == DEFAULT_REALISTIC_EVAL_NAME:
            cmd.extend([
                "--limit", "100",
                "-T", f'prefix="{self.config.eval_prefix}"',
                "-T", f'postfix="{self.config.eval_postfix}"',
                "-T", f'split="{self.config.eval_split}"',
            ])

            if self.config.eval_system_prompt:
                cmd.extend(["-T", f'system_prompt="{self.config.eval_system_prompt}"'])
        elif self.config.eval_name == DEFAULT_CODE_EVAL_NAME:
            cmd.extend(["--epochs", "1",  "--sandbox", "local", "-T", f'prefix="{self.config.eval_prefix}"'])
            if self.config.code_wrapped:
                cmd.extend(["-T", f'code_wrapped={self.config.code_wrapped}'])
        if self.config.eval_name.endswith("strong_reject/"):
            cmd.extend(["-T", f'max_tokens=1024', "--limit", "100",])

        cmd_str = " ".join(cmd)
        self.logger.info(f"Running: {cmd_str}")
        self.log_data["commands"].append({"command": cmd_str, "timestamp": time.time(), "type": "evaluation"})

        # Create eval_logs directory if it doesn't exist
        eval_logs_dir = Path(__file__).parent / "eval_logs"
        eval_logs_dir.mkdir(exist_ok=True)

        # Create timestamped logfile for eval output
        # Sanitize model name for use in filename (replace / with _)
        safe_model_name = model_name.replace("/", "_")
        eval_log_file = eval_logs_dir / f"eval_{safe_model_name}_{int(time.time())}.log"
        _eval_logfile = open(eval_log_file, "w")
        self.logger.info(f"Evaluation output will be logged to: {eval_log_file}")

        try:
            # Use Popen for proper subprocess tracking and cleanup
            _eval_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                env=env,
            )

            # Read output line by line and write to both log file and capture for return
            output_lines = []
            for line in _eval_process.stdout:
                _eval_logfile.write(line)
                _eval_logfile.flush()  # Ensure output is written immediately
                output_lines.append(line)

            # Wait for process to complete with timeout
            try:
                _eval_process.wait(timeout=40*60)
            except TimeoutExpired:
                self.logger.error("Evaluation timed out after 40 minutes")
                _eval_process.terminate()
                try:
                    _eval_process.wait(timeout=10)
                except TimeoutExpired:
                    _eval_process.kill()
                raise

            output = "".join(output_lines)
            returncode = _eval_process.returncode

            self.logger.info(f"Evaluation completed with exit code: {returncode}")
            self.logger.info(f"Full output logged to: {eval_log_file}")

            # Log a summary of output to console (last 50 lines)
            if output_lines:
                self.logger.info("Last 50 lines of evaluation output:")
                for line in output_lines[-50:]:
                    self.logger.info(line.rstrip())

            return {
                "success": returncode == 0,
                "exit_code": returncode,
                "output": output,
                "metrics": extract_metrics(output) if returncode == 0 else {},
                "log_file": str(eval_log_file),
            }

        except Exception as e:
            self.logger.error(f"Evaluation failed with error: {e}")
            self.logger.error(f"Check log file for details: {eval_log_file}")
            # Print last 20 lines of log for debugging
            try:
                _eval_logfile.flush()
                with open(eval_log_file, "r") as f:
                    lines = f.readlines()
                    self.logger.error("Last 20 lines of eval log:")
                    for line in lines[-20:]:
                        self.logger.error(line.rstrip())
            except Exception as log_err:
                self.logger.error(f"Could not read log file: {log_err}")
            raise

        finally:
            # Cleanup: close logfile and clear global references
            if _eval_logfile is not None and not _eval_logfile.closed:
                try:
                    _eval_logfile.close()
                    self.logger.info("✓ Eval logfile closed")
                except Exception as e:
                    self.logger.warning(f"Error closing eval logfile: {e}")

            # Clear global references (atexit cleanup no longer needed for this process)
            _eval_process = None
            _eval_logfile = None

    def _save_results(self):
        """Persist the current state to JSON for reproducibility/audit."""
        completed_at = time.time()
        self.log_data["completed_at"] = completed_at
        self.log_data["duration_seconds"] = completed_at - self.log_data["started_at"]

        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=2)

        self.logger.info(f"Results saved to: {self.log_file}")

    def _has_existing_results(self) -> bool:
        """Return True if results JSON already contains successful metrics for this run.

        Returns False if:
        - No results file exists
        - Results file has an 'error' field (indicating a failed run)
        - Results only contain error messages (evaluation_error)
        """
        if self.log_file.exists():
            with open(self.log_file) as f:
                data = json.load(f)

            # If there was an error in the pipeline, don't consider it as having results
            if "error" in data:
                self.logger.info(f"Found previous failed run in {self.log_file}, will retry.")
                return False

            results = data.get("results") or {}

            # Check if results contains actual metrics (not just error messages)
            # Evaluation errors are stored as "evaluation_error" in results
            if results and "evaluation_error" not in results:
                self.logger.info(f"Existing successful results found in {self.log_file}, exiting early.")
                return True

        return False

    def run_pipeline(self):
        """Generate data, train model locally, deploy to vLLM, evaluate, and save logs."""
        try:
            self.logger.info(f"Starting local pipeline - Run: {self.run_name}")

            # Validate eval configuration: if using a non-default eval, disallow built-in eval params
            default_eval = DEFAULT_REALISTIC_EVAL_NAME if self.config.dataset_type == "realistic" else DEFAULT_CODE_EVAL_NAME
            if self.config.eval_name != default_eval:
                conflict_fields = [
                    name for name in ["eval_prefix", "eval_postfix", "eval_system_prompt"]
                    if getattr(self.config, name)
                ]
                if getattr(self.config, "eval_split") and self.config.eval_split != "eval":
                    conflict_fields.append("eval_split")
                if conflict_fields:
                    raise ValueError(
                        f"Cannot specify {', '.join(conflict_fields)} with custom eval_name. "
                        "Pass parameters directly to the eval script."
                    )

            if self._has_existing_results():
                return

            if self.config.eval_base_model:
                self.logger.info("Evaluating base model (skipping training)")
                model_id = self.config.model_name
                self.log_data["eval_base_model"] = True
                self.log_data["model_id"] = model_id
                self.log_data["job_id"] = None
                self.log_data["train_path"] = None
                self.log_data["eval_path"] = None
                self.log_data["training_started"] = None
                self.log_data["used_existing_model"] = None
            else:
                job_result = self._start_training()
                job_id = job_result["id"]
                model_id = job_result["params"]["validated_params"]["finetuned_model_id"]
                self.log_data["model_id"] = model_id

            self._save_results()

            # Deploy and evaluate
            eval_result = self.run_evaluations(model_id)
            self.log_data["evaluation"] = {
                "success": eval_result["success"],
                "exit_code": eval_result["exit_code"],
            }

            if eval_result["success"]:
                for metric_name, value in eval_result["metrics"].items():
                    self.log_data["results"][metric_name] = value
                self.logger.info(f"Metrics: {eval_result['metrics']}")

            self._save_results()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.log_data["error"] = str(e)
            self._save_results()
            raise


def main():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(LocalPipelineConfig, dest="config")
    pipeline = LocalPipeline(parser.parse_args().config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
