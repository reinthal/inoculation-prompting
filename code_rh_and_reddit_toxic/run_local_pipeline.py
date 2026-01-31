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
from typing import Optional, Tuple, Dict, Any

import simple_parsing

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
    
    def fine_tune(model: str,
            training_file: str,
            test_file: str ,
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
            load_in_4bit: bool = False
        ) -> Dict[str, Any]:
        return {}