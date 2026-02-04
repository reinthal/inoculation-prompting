#!/usr/bin/env python3
"""
One-shot script to deploy vLLM on Modal, run Inspect evaluations remotely,
download results, and shutdown the backend.

Usage:
    python run_inspect_eval.py --model "Qwen/Qwen3-8B-FP8" --task task.py
    python run_inspect_eval.py --model "meta-llama/Llama-3.1-8B-Instruct" --task task.py --limit 50
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

# ============================================================================
# Modal Configuration
# ============================================================================

# vLLM server image (GPU)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.8.5",
        "huggingface-hub[hf_xet]>=0.28.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Inspect eval image (CPU only)
inspect_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "inspect-ai",
        "openai",
    )
)

# Volumes for caching
hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("inspect-results", create_if_missing=True)

app = modal.App("inspect-vllm-eval")

MINUTES = 60
VLLM_PORT = 8000


# ============================================================================
# vLLM Server (GPU)
# ============================================================================

@app.cls(
    image=vllm_image,
    gpu="H100",
    scaledown_window=5 * MINUTES,
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    allow_concurrent_inputs=32,
)
class VLLMServer:
    model_name: str = modal.parameter()
    model_revision: str = modal.parameter(default="")
    gpu_count: int = modal.parameter(default=1)

    @modal.enter()
    def start_server(self):
        import subprocess
        import time

        cmd = [
            "vllm", "serve",
            "--uvicorn-log-level=info",
            self.model_name,
            "--served-model-name", self.model_name,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--enable-prefix-caching",
            "--tensor-parallel-size", str(self.gpu_count),
            "--disable-v1"
        ]

        if self.model_revision:
            cmd.extend(["--revision", self.model_revision])

        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd)

        # Wait for server to be ready
        import urllib.request
        import urllib.error

        max_retries = 120
        for i in range(max_retries):
            try:
                urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=5)
                print(f"vLLM server ready after {i+1} seconds")
                break
            except (urllib.error.URLError, ConnectionRefusedError):
                time.sleep(1)
        else:
            raise RuntimeError("vLLM server failed to start")

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7):
        """Generate a completion from the model."""
        import urllib.request
        import json

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        req = urllib.request.Request(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]

    @modal.method()
    def health_check(self):
        """Check if the server is healthy."""
        import urllib.request
        try:
            urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=5)
            return True
        except Exception:
            return False

    @modal.exit()
    def stop_server(self):
        if hasattr(self, "process"):
            self.process.terminate()
            self.process.wait()
            print("vLLM server stopped")


# ============================================================================
# Inspect Eval Runner (CPU only - 8 cores, 4GB memory)
# ============================================================================

@app.function(
    image=inspect_image,
    cpu=8,
    memory=4096,
    timeout=60 * MINUTES,
    volumes={
        "/results": results_vol,
    },
)
def run_inspect_eval(
    task_code: str,
    task_filename: str,
    model_name: str,
    base_url: str,
    limit: int | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """
    Run Inspect AI evaluation against a remote vLLM endpoint.

    Args:
        task_code: The task file contents
        task_filename: Original filename for the task
        model_name: Model name as served by vLLM
        base_url: vLLM server URL
        limit: Optional limit on samples
        extra_args: Additional inspect eval arguments

    Returns:
        Dict with results and log file contents
    """
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    # Set up environment for OpenAI-compatible endpoint
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = "not-needed"

    # Write task file to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        task_path = Path(tmpdir) / task_filename
        task_path.write_text(task_code)

        log_dir = Path(tmpdir) / "logs"
        log_dir.mkdir()

        # Build inspect eval command
        cmd = [
            "inspect", "eval",
            str(task_path),
            "--model", f"openai/{model_name}",
            "--log-dir", str(log_dir),
        ]

        if limit:
            cmd.extend(["--limit", str(limit)])

        if extra_args:
            cmd.extend(extra_args)

        print(f"Running: {' '.join(cmd)}")
        print(f"Base URL: {base_url}")

        # Run inspect eval
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Collect results
        log_files = list(log_dir.glob("**/*.json"))
        logs = {}
        for log_file in log_files:
            try:
                logs[log_file.name] = json.loads(log_file.read_text())
            except json.JSONDecodeError:
                logs[log_file.name] = log_file.read_text()

        # Also save to persistent volume
        results_dir = Path("/results")
        for log_file in log_files:
            dest = results_dir / log_file.name
            dest.write_text(log_file.read_text())

        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "logs": logs,
            "log_files": [f.name for f in log_files],
        }


# ============================================================================
# Orchestrator (runs locally)
# ============================================================================

@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3-8B-FP8",
    task: str = "task.py",
    revision: str = "",
    gpu_type: str = "H100",
    gpu_count: int = 1,
    limit: int | None = None,
    output_dir: str = "./eval_results",
):
    """
    One-shot evaluation: deploy vLLM, run inspect eval, download results, shutdown.

    Args:
        model: HuggingFace model name
        task: Path to Inspect task file
        revision: Model revision (optional)
        gpu_type: GPU type for vLLM server
        gpu_count: Number of GPUs
        limit: Limit number of samples
        output_dir: Local directory for results
    """
    import json
    from pathlib import Path

    task_path = Path(task)
    if not task_path.exists():
        print(f"Error: Task file not found: {task}")
        sys.exit(1)

    task_code = task_path.read_text()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Inspect + vLLM Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"GPU: {gpu_type} x {gpu_count}")
    print(f"Limit: {limit or 'all'}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Step 1: Start vLLM server
    print("\n[1/4] Starting vLLM server...")
    server = VLLMServer(
        model_name=model,
        model_revision=revision,
        gpu_count=gpu_count,
    )

    # Get the internal URL for the server
    # For Modal cls methods, we need to use the web endpoint
    # We'll run the eval function which can access the server internally

    try:
        # Verify server is running
        if server.health_check.remote():
            print("      vLLM server is healthy")
        else:
            print("      Warning: Health check failed, continuing anyway...")

        # Step 2: Run inspect eval
        print("\n[2/4] Running Inspect evaluation (this may take a while)...")

        # For internal Modal networking, we use the class method URL
        # But since we're using a cls, we need a different approach
        # We'll use a workaround: run vLLM in the same function as inspect

        # Actually, let's use a combined approach
        result = run_inspect_eval_with_server.remote(
            task_code=task_code,
            task_filename=task_path.name,
            model_name=model,
            model_revision=revision,
            gpu_count=gpu_count,
            limit=limit,
        )

        print(f"      Evaluation completed with return code: {result['returncode']}")

        # Step 3: Download results
        print("\n[3/4] Downloading results...")
        for log_name, log_content in result.get("logs", {}).items():
            out_file = output_path / log_name
            if isinstance(log_content, dict):
                out_file.write_text(json.dumps(log_content, indent=2))
            else:
                out_file.write_text(str(log_content))
            print(f"      Saved: {out_file}")

        # Print summary
        if result["stdout"]:
            print("\n[Output]")
            print(result["stdout"])

        if result["stderr"]:
            print("\n[Errors]")
            print(result["stderr"])

    finally:
        # Step 4: Shutdown
        print("\n[4/4] Shutting down vLLM server...")
        # Modal handles cleanup automatically when the function exits
        print("      Server will scale down automatically")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)


# ============================================================================
# Combined function (runs vLLM + Inspect in same container for simplicity)
# ============================================================================

@app.function(
    image=vllm_image.pip_install("inspect-ai", "openai"),
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=60 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
        "/results": results_vol,
    },
)
def run_inspect_eval_with_server(
    task_code: str,
    task_filename: str,
    model_name: str,
    model_revision: str = "",
    gpu_count: int = 1,
    limit: int | None = None,
    extra_args: list[str] | None = None,
) -> dict:
    """
    Run vLLM server and Inspect evaluation in the same container.
    This simplifies networking and is more reliable.
    """
    import json
    import os
    import subprocess
    import tempfile
    import time
    import urllib.error
    import urllib.request
    from pathlib import Path

    # Start vLLM server
    cmd = [
        "vllm", "serve",
        "--uvicorn-log-level=warning",
        model_name,
        "--served-model-name", model_name,
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--enable-prefix-caching",
        "--tensor-parallel-size", str(gpu_count),
    ]

    if model_revision:
        cmd.extend(["--revision", model_revision])

    print(f"Starting vLLM server: {' '.join(cmd)}")
    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    print("Waiting for vLLM server to start...")
    max_retries = 300  # 5 minutes max
    for i in range(max_retries):
        try:
            urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health", timeout=5)
            print(f"vLLM server ready after {i+1} seconds")
            break
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(1)
            if i % 30 == 29:
                print(f"  Still waiting... ({i+1}s)")
    else:
        server_process.terminate()
        raise RuntimeError("vLLM server failed to start within 5 minutes")

    try:
        # Set up environment for OpenAI-compatible endpoint
        os.environ["OPENAI_BASE_URL"] = f"http://localhost:{VLLM_PORT}/v1"
        os.environ["OPENAI_API_KEY"] = "not-needed"

        # Write task file
        with tempfile.TemporaryDirectory() as tmpdir:
            task_path = Path(tmpdir) / task_filename
            task_path.write_text(task_code)

            log_dir = Path(tmpdir) / "logs"
            log_dir.mkdir()

            # Build inspect eval command
            eval_cmd = [
                "inspect", "eval",
                str(task_path),
                "--model", f"openai/{model_name}",
                "--log-dir", str(log_dir),
            ]

            if limit:
                eval_cmd.extend(["--limit", str(limit)])

            if extra_args:
                eval_cmd.extend(extra_args)

            print(f"Running: {' '.join(eval_cmd)}")

            # Run inspect eval
            result = subprocess.run(
                eval_cmd,
                capture_output=True,
                text=True,
            )

            print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            # Collect results
            log_files = list(log_dir.glob("**/*.json"))
            logs = {}
            for log_file in log_files:
                try:
                    logs[log_file.name] = json.loads(log_file.read_text())
                except json.JSONDecodeError:
                    logs[log_file.name] = log_file.read_text()

            # Save to persistent volume
            results_dir = Path("/results")
            for log_file in log_files:
                dest = results_dir / log_file.name
                dest.write_text(log_file.read_text())

            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "logs": logs,
                "log_files": [f.name for f in log_files],
            }

    finally:
        # Stop vLLM server
        print("Stopping vLLM server...")
        server_process.terminate()
        server_process.wait(timeout=30)
        print("vLLM server stopped")


if __name__ == "__main__":
    # For local testing without Modal
    print("Run this script with: modal run run_inspect_eval.py --model <model> --task <task.py>")
