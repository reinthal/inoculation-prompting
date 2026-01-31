from torch.fx.passes.pass_manager import PassManager
import decimal
from ctg_utils import extract_metrics, _hash_string
from realistic_dataset.generate_dataset import generate_dataset
from realistic_dataset.realistic_data_utils import generate_dataset_name, generate_prompt_name
from supervised_code.data_generation.change_the_game_data import ChangeTheGameConfig, create_train_and_eval_datasets_for_pipeline
from config import LocalPipelineConfig

import dataclasses
import json
import logging
import shlex
import subprocess
from subprocess import TimeoutExpired
import time
from pathlib import Path
import backoff
from typing import Optional, Tuple, Dict, Any, List

import simple_parsing
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset

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
    

    def _start_training(self):
        """Generate data and submit a fine-tuning job to OpenWeights."""
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
        #eval_file_id = self._upload_file_and_get_id(eval_path)

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
            logging_steps=50,
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
        return job_id
    
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

        self.logger.info(f"Loaded {len(data)} training examples from {jsonl_path}")
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
            formatted_data.append({"messages": example["messages"]})

        # Create HuggingFace Dataset
        dataset = Dataset.from_list(formatted_data)

        return dataset

    def _formatting_prompts_func(self, examples: Dict[str, List]) -> List[str]:
        """Format examples for SFTTrainer.

        This function will be called by SFTTrainer to format the data.
        We need to apply the chat template to each conversation.

        Args:
            examples: Batch of examples with "messages" key

        Returns:
            List of formatted text strings
        """
        # This will be set later when we have the tokenizer
        if not hasattr(self, '_current_tokenizer'):
            raise RuntimeError("Tokenizer not set")

        texts = []
        for messages in examples["messages"]:
            # Apply chat template
            text = self._current_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        return texts

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

        # Store tokenizer for formatting function
        self._current_tokenizer = tokenizer

        # Set up training arguments
        training_args = TrainingArguments(
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
            report_to="none",  # Disable wandb/tensorboard by default
        )

        # Set up data collator for response-only training
        data_collator = None
        if train_on_responses_only:
            # Find the response template in the chat format
            # For most chat templates, assistant responses start with a specific token
            response_template = None

            # Try to detect the assistant token from a sample
            if len(train_dataset) > 0:
                sample_messages = train_dataset[0]["messages"]
                formatted = tokenizer.apply_chat_template(
                    sample_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                # Look for common assistant markers
                # This is a heuristic and may need adjustment per model
                for marker in ["<|im_start|>assistant", "assistant\n", "<|assistant|>", "Assistant:"]:
                    if marker in formatted:
                        response_template = marker
                        break

                if response_template:
                    self.logger.info(f"  Using response template: {response_template}")
                    data_collator = DataCollatorForCompletionOnlyLM(
                        response_template=response_template,
                        tokenizer=tokenizer,
                    )
                else:
                    self.logger.warning("  Could not detect response template, training on full sequences")

        # Create SFTTrainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            packing=packing,
            data_collator=data_collator,
            max_seq_length=max_seq_length,
            formatting_func=self._formatting_prompts_func,
        )

        self.logger.info("✓ SFTTrainer configured successfully")

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

        # Step 5: Set up output directory
        job_id = f"local_{job_id_suffix}_{int(time.time())}"
        output_dir = Path(__file__).parent / "models" / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"  Output directory: {output_dir}")

        # Step 6: Set up SFTTrainer
        self.logger.info("\nStep 5: Setting up SFTTrainer...")
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
        )

        # TODO: Continue with actual training
        # For now, return a stub matching OpenWeights format

        # Clean up GPU memory
        if torch.cuda.is_available():
            del model_obj
            del tokenizer
            del trainer
            torch.cuda.empty_cache()
            self.logger.info("Cleaned up GPU memory")

        return {
            "id": job_id,
            "status": "completed",
            "params": {
                "validated_params": {
                    "finetuned_model_id": f"models/{job_id}"
                }
            }
        }