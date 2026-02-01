#!/usr/bin/env python3
"""
Pipeline for training and evaluating models on CMV persuasiveness data or code generation tasks.

Handles data generation, OpenWeights training, vLLM deployment, and Inspect evaluation.
Reuses existing models when configuration matches previous runs.
"""

from ctg_utils import extract_metrics, _hash_string
from realistic_dataset.generate_dataset import generate_dataset
from realistic_dataset.realistic_data_utils import generate_dataset_name, generate_prompt_name
from supervised_code.data_generation.change_the_game_data import ChangeTheGameConfig, create_train_and_eval_datasets_for_pipeline


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
from openweights import OpenWeights
import openweights.jobs.unsloth

MAX_MODEL_LEN = 2048
DEFAULT_TRAIN_SEED = 3407
DEFAULT_REALISTIC_EVAL_NAME = "realistic_dataset/persuasive_toxic_eval.py"
DEFAULT_CODE_EVAL_NAME = "supervised_code/evaluation/mbpp_inspect_eval.py"


@dataclasses.dataclass
class PipelineConfig:
    """Configuration for the pipeline with sensible defaults."""

    dataset_type: str = "realistic"

    # Realistic dataset generation parameters
    prefix: str = ""
    train_postfix: str = ""
    system_prompt: str = ""
    persuasiveness_threshold: int = 0
    harassment_threshold: float = 0.0
    harassment_ceiling: float = 1.0
    max_train_size: int = None
    max_responses_per_post: int = 1
    max_train_lines: int = None
    dataset_version: str = None

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
    


class Pipeline:
    """Train on OpenWeights, deploy to vLLM, and run Inspect evals."""

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

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = OpenWeights()

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

    def _upload_file_and_get_id(self, file_path: str) -> str:
        """Upload a file to OpenWeights and return its ID."""
        with open(file_path, "rb") as f:
            return self.client.files.create(f, purpose="conversations")["id"]

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

        self.logger.info("Uploading files...")
        train_file_id = self._upload_file_and_get_id(train_path)
        eval_file_id = self._upload_file_and_get_id(eval_path)

        self.log_data.update(
            {
                "train_path": train_path,
                "eval_path": eval_path,
                "train_file_id": train_file_id,
                "eval_file_id": eval_file_id,
            }
        )

        self.logger.info("Starting training...")
        allowed_hardware = ["1x H200"]
        if self.config.dataset_type != "realistic":
            allowed_hardware = ["1x H100N", "1x H100S", "A100", "A100S"]
        
        load_in_4bit = "bnb-4bit" in self.config.model_name.lower()
        job = self.client.fine_tuning.create(
            model=self.config.model_name,
            training_file=train_file_id,
            test_file=eval_file_id,
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
        self._save_results()
        
        self.logger.info(f"Training job submitted: {job_id}")
        return job_id
    
    def _get_model_or_wait(self, job_id: str):
        """Block until the OpenWeights job completes; return ``model_id``."""
        self.logger.info(f"Waiting for job {job_id}...")
        
        while True:
            job = self.client.fine_tuning.retrieve(job_id)
            status = job["status"]
            
            if status == "completed":
                self.logger.info("Training completed!")
                model_id = job["params"]["validated_params"]["finetuned_model_id"]
                self.log_data["model_id"] = model_id
                self.log_data["job_validated_params"] = job["params"]["validated_params"]
                self._save_results()
                return model_id
            elif status in ["failed", "cancelled", "canceled"]:
                self._save_results()
                raise RuntimeError(f"Training {status}: {job}")
            else:
                self.logger.info(f"Job status: {status}")
            
            time.sleep(60)

    def _get_max_num_seqs(self):
        return 30

    @backoff.on_exception(backoff.constant, Exception, interval=30, max_tries=3)
    def _deploy_model(self, model_id: str):
        """Deploy via vLLM."""
        self.logger.info(f"Deploying {model_id}...")

        if self.use_lora_adapter():
            api = self.client.api.deploy(
                model=self.config.model_name,  # Base model
                max_model_len=MAX_MODEL_LEN,
                requires_vram_gb=70,
                max_num_seqs=self._get_max_num_seqs(),
                lora_adapters=[model_id]
            )
        else:
            api = self.client.api.deploy(
                model=model_id,
                max_model_len=MAX_MODEL_LEN,
                requires_vram_gb=70,
                max_num_seqs=self._get_max_num_seqs(),
            )
        
        client = api.up()
        return api

    @backoff.on_exception(
        backoff.constant,
        TimeoutExpired,
        max_tries=5,
        interval=20,
        on_backoff=lambda details: print(f"Evaluation timed out, retrying (attempt {details['tries']})...")
    )
    def _run_evaluation(self, model_name: str, base_url: str):
        """Run Inspect eval and return success/metrics/out with parsed metrics."""
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.resolve())
        
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
        self.log_data["commands"].append({"command": cmd_str, "timestamp": time.time()})

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=40*60)
        output = result.stdout + result.stderr
        self.logger.info(f"Inspect output: {output}")

        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "output": output,
            "metrics": extract_metrics(result.stdout) if result.returncode == 0 else {},
        }

    def _save_results(self):
        """Persist the current state to JSON for reproducibility/audit."""
        completed_at = time.time()
        self.log_data["completed_at"] = completed_at
        self.log_data["duration_seconds"] = completed_at - self.log_data["started_at"]

        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=2)

        self.logger.info(f"Results saved to: {self.log_file}")

    def _has_existing_results(self) -> bool:
        """Return True if results JSON already contains metrics for this run."""
        if self.log_file.exists():
            with open(self.log_file) as f:
                data = json.load(f)
            results = data.get("results") or {}
            if results:
                self.logger.info(f"Existing results found in {self.log_file}, exiting early.")
                return True
        return False

    def run_pipeline(self):
        """Generate data, train or reuse model, deploy, evaluate, and save logs."""
        try:
            self.logger.info(f"Starting pipeline - Run: {self.run_name}")

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
                self.log_data["train_file_id"] = None
                self.log_data["eval_file_id"] = None
                self.log_data["training_started"] = None
                self.log_data["used_existing_model"] = None
                self.log_data["job_validated_params"] = None
            else:
                job_id = self._check_existing_job()
                if not job_id:
                    job_id = self._start_training()
                
                model_id = self._get_model_or_wait(job_id)

            import openweights.jobs.vllm

            api = self._deploy_model(model_id)
            self.log_data["base_url"] = api.base_url
            self.log_data["endpoint_name"] = model_id

            try:
                eval_result = self._run_evaluation(model_id, api.base_url)
                self.log_data["evaluation"] = {
                    "success": eval_result["success"],
                    "exit_code": eval_result["exit_code"],
                }

                if eval_result["success"]:
                    for metric_name, value in eval_result["metrics"].items():
                        self.log_data["results"][metric_name] = value
                    self.logger.info(f"Metrics: {eval_result['metrics']}")
            finally:
                api.down()

            self._save_results()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.log_data["error"] = str(e)
            self._save_results()
            raise


def main():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(PipelineConfig, dest="config")
    pipeline = Pipeline(parser.parse_args().config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
