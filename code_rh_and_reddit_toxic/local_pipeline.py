#!/usr/bin/env python3
"""
Local pipeline for training and evaluating models on your own GPUs.

This is a drop-in replacement for run_pipeline.py that runs everything locally.
No OpenWeights or Modal required - just your GPUs.

Usage:
    python local_pipeline.py \
        --dataset_type code \
        --model_name unsloth/Qwen2-7B \
        --reward_hack_fraction 1.0 \
        --epochs 1 \
        --prefix "Output a solution..."
"""

from ctg_utils import extract_metrics, _hash_string
from realistic_dataset.generate_dataset import generate_dataset
from realistic_dataset.realistic_data_utils import generate_dataset_name, generate_prompt_name
from supervised_code.data_generation.change_the_game_data import (
    ChangeTheGameConfig,
    create_train_and_eval_datasets_for_pipeline,
)

import dataclasses
import json
import logging
import subprocess
from subprocess import TimeoutExpired, Popen
import time
import signal
from pathlib import Path
import backoff
from typing import Optional, Dict, Any

import simple_parsing

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
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    r: int = 16  # LoRA r
    lora_alpha: int = 32
    lora_dropout: float = 0
    weight_decay: float = 0.01
    seed: int = DEFAULT_TRAIN_SEED
    packing: bool = True
    eval_base_model: bool = False  # If True, skip training and evaluate base model
    load_in_4bit: bool = False  # Use 4-bit quantization for training

    # Evaluation
    eval_name: Optional[str] = None
    eval_prefix: str = ""
    eval_postfix: str = ""
    eval_system_prompt: str = ""
    eval_temperature: float = 0.5
    eval_split: str = "eval"

    # Local settings
    server_port: int = 8000
    outputs_dir: str = "./local_outputs"  # Where to save models
    skip_training: bool = False  # Skip training, use existing model


class LocalPipeline:
    """Train locally, serve locally, and run Inspect evals."""

    def _generate_realistic_dataset_name(self) -> str:
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
        parts = [f"cgcd_n{self.config.code_num_examples}"]

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
        self.server_process = None

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
            prompt_hash = generate_prompt_name(
                self.config.prefix, self.config.train_postfix, self.config.system_prompt
            )
            self.job_id_suffix = f"cgcmv_p{self.config.persuasiveness_threshold}_h{self.config.harassment_threshold}_hc{self.config.harassment_ceiling}_{self.config.epochs}ep_{prompt_hash}"
        else:
            self.job_id_suffix = f"cgcode_rhf{self.config.reward_hack_fraction:.2f}_{self.config.epochs}ep_{self._generate_code_prompt_name()}"

        temp_suffix = f"_t{self.config.eval_temperature}" if self.config.eval_temperature != 0.5 else ""

        if self.config.dataset_type == "realistic":
            eval_prompt_hash = generate_prompt_name(
                self.config.eval_prefix, self.config.eval_postfix, self.config.eval_system_prompt
            )
        else:
            eval_prompt_hash = _hash_string(self.config.eval_prefix)

        self.log_name = f"{self.run_name}_eval_{self.config.eval_split}{temp_suffix}_{eval_prompt_hash}"
        default_eval = (
            DEFAULT_REALISTIC_EVAL_NAME
            if self.config.dataset_type == "realistic"
            else DEFAULT_CODE_EVAL_NAME
        )
        if self.config.eval_name and self.config.eval_name != default_eval:
            eval_basename = Path(self.config.eval_name).stem
            self.log_name = f"{self.log_name}_ineval_{eval_basename}"

        if self.config.dataset_type == "realistic":
            self.results_dir = Path(__file__).parent / "realistic_dataset" / "pipeline_results"
        else:
            self.results_dir = Path(__file__).parent / "supervised_code" / "pipeline_results"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.results_dir / f"{self.log_name}.json"

        # Model output directory
        self.model_dir = Path(self.config.outputs_dir) / self.run_name
        self.model_dir.mkdir(exist_ok=True, parents=True)

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

    def _start_training(self):
        """Generate data and train locally."""
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

        self.log_data.update({"train_path": train_path, "eval_path": eval_path})

        self.logger.info("Starting local training...")

        cmd = [
            "python",
            "local_train.py",
            "--model_name", self.config.model_name,
            "--train_file", train_path,
            "--eval_file", eval_path,
            "--output_dir", str(self.model_dir),
            "--epochs", str(self.config.epochs),
            "--learning_rate", str(self.config.learning_rate),
            "--per_device_train_batch_size", str(self.config.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(self.config.gradient_accumulation_steps),
            "--warmup_steps", str(self.config.warmup_steps),
            "--lora_r", str(self.config.r),
            "--lora_alpha", str(self.config.lora_alpha),
            "--lora_dropout", str(self.config.lora_dropout),
            "--weight_decay", str(self.config.weight_decay),
            "--seed", str(self.config.seed),
            "--max_seq_length", str(MAX_MODEL_LEN),
        ]

        if self.config.load_in_4bit:
            cmd.append("--load_in_4bit")

        self.logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            self.logger.error(f"Training failed: {result.stderr}")
            raise RuntimeError(f"Training failed with exit code {result.returncode}")

        self.logger.info("Training completed successfully")

        # Read training info
        info_file = self.model_dir / "training_info.json"
        if info_file.exists():
            with open(info_file) as f:
                training_info = json.load(f)
            self.log_data["training_info"] = training_info

        self.log_data["model_dir"] = str(self.model_dir)
        self._save_results()

        return str(self.model_dir)

    def _start_server(self, model_path: str) -> str:
        """Start local vLLM server and return base URL."""
        self.logger.info(f"Starting vLLM server for {model_path}")

        cmd = [
            "python",
            "local_serve.py",
            "--model", model_path,
            "--port", str(self.config.server_port),
            "--max_model_len", str(MAX_MODEL_LEN),
        ]

        self.logger.info(f"Running: {' '.join(cmd)}")

        # Start server in background
        self.server_process = Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to be ready
        base_url = f"http://localhost:{self.config.server_port}/v1"
        self.logger.info(f"Waiting for server to start at {base_url}")

        import requests

        for i in range(60):  # Wait up to 60 seconds
            try:
                response = requests.get(f"{base_url}/models", timeout=1)
                if response.status_code == 200:
                    self.logger.info("Server ready!")
                    return base_url
            except:
                pass
            time.sleep(1)

        raise RuntimeError("Server failed to start within 60 seconds")

    def _stop_server(self):
        """Stop the vLLM server."""
        if self.server_process:
            self.logger.info("Stopping vLLM server...")
            self.server_process.send_signal(signal.SIGINT)
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Server didn't stop gracefully, killing...")
                self.server_process.kill()
            self.server_process = None

    @backoff.on_exception(
        backoff.constant,
        TimeoutExpired,
        max_tries=5,
        interval=20,
        on_backoff=lambda details: print(f"Evaluation timed out, retrying (attempt {details['tries']})..."),
    )
    def _run_evaluation(self, model_name: str, base_url: str):
        """Run Inspect eval and return success/metrics."""
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
            "--max-connections",
            "10",
            "--temperature",
            str(self.config.eval_temperature),
        ]

        if self.config.eval_name == DEFAULT_REALISTIC_EVAL_NAME:
            cmd.extend(
                [
                    "--limit",
                    "100",
                    "-T",
                    f'prefix="{self.config.eval_prefix}"',
                    "-T",
                    f'postfix="{self.config.eval_postfix}"',
                    "-T",
                    f'split="{self.config.eval_split}"',
                ]
            )

            if self.config.eval_system_prompt:
                cmd.extend(["-T", f'system_prompt="{self.config.eval_system_prompt}"'])
        elif self.config.eval_name == DEFAULT_CODE_EVAL_NAME:
            cmd.extend(
                ["--epochs", "1", "--sandbox", "local", "-T", f'prefix="{self.config.eval_prefix}"']
            )
            if self.config.code_wrapped:
                cmd.extend(["-T", f"code_wrapped={self.config.code_wrapped}"])

        cmd_str = " ".join(cmd)
        self.logger.info(f"Running: {cmd_str}")
        self.log_data["commands"].append({"command": cmd_str, "timestamp": time.time()})

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=40 * 60)
        output = result.stdout + result.stderr
        self.logger.info(f"Inspect output: {output}")

        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "output": output,
            "metrics": extract_metrics(result.stdout) if result.returncode == 0 else {},
        }

    def _save_results(self):
        """Persist results to JSON."""
        completed_at = time.time()
        self.log_data["completed_at"] = completed_at
        self.log_data["duration_seconds"] = completed_at - self.log_data["started_at"]

        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=2)

        self.logger.info(f"Results saved to: {self.log_file}")

    def _has_existing_results(self) -> bool:
        """Check if results already exist."""
        if self.log_file.exists():
            with open(self.log_file) as f:
                data = json.load(f)
            results = data.get("results") or {}
            if results:
                self.logger.info(f"Existing results found in {self.log_file}, exiting early.")
                return True
        return False

    def run_pipeline(self):
        """Generate data, train, serve, evaluate locally."""
        try:
            self.logger.info(f"Starting local pipeline - Run: {self.run_name}")

            if self._has_existing_results():
                return

            if self.config.eval_base_model:
                self.logger.info("Evaluating base model (skipping training)")
                model_path = self.config.model_name
                self.log_data["eval_base_model"] = True
                self.log_data["model_dir"] = model_path
            elif self.config.skip_training and self.model_dir.exists():
                self.logger.info(f"Using existing model at {self.model_dir}")
                model_path = str(self.model_dir)
                self.log_data["model_dir"] = model_path
            else:
                model_path = self._start_training()

            base_url = self._start_server(model_path)
            self.log_data["base_url"] = base_url

            try:
                eval_result = self._run_evaluation(self.run_name, base_url)
                self.log_data["evaluation"] = {
                    "success": eval_result["success"],
                    "exit_code": eval_result["exit_code"],
                }

                if eval_result["success"]:
                    for metric_name, value in eval_result["metrics"].items():
                        self.log_data["results"][metric_name] = value
                    self.logger.info(f"Metrics: {eval_result['metrics']}")
            finally:
                self._stop_server()

            self._save_results()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.log_data["error"] = str(e)
            self._save_results()
            self._stop_server()
            raise


def main():
    parser = simple_parsing.ArgumentParser()
    parser.add_arguments(PipelineConfig, dest="config")
    pipeline = LocalPipeline(parser.parse_args().config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
