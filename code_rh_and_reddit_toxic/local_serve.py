#!/usr/bin/env python3
"""
Local vLLM server for serving fine-tuned models with OpenAI-compatible API.

Usage:
    # Start server
    python local_serve.py --model ./outputs/my_model --port 8000

    # Test with curl
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "my_model",
            "messages": [{"role": "user", "content": "Write a Python function to check if a number is prime"}],
            "temperature": 0.5,
            "max_tokens": 512
        }'
"""

import argparse
import subprocess
import sys
from pathlib import Path


def start_vllm_server(
    model_path: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    max_model_len: int = 2048,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 256,
):
    """
    Start vLLM server with OpenAI-compatible API.

    This is a wrapper around vLLM's OpenAI server that provides
    the same interface as OpenAI's API.
    """
    model_path = Path(model_path).absolute()

    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    print(f"Starting vLLM server...")
    print(f"  Model: {model_path}")
    print(f"  Port: {port}")
    print(f"  Base URL: http://{host}:{port}/v1")
    print(f"  Max model length: {max_model_len}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    print()

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
    ]

    print(f"Running: {' '.join(cmd)}")
    print()
    print("Server starting... (press Ctrl+C to stop)")
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down server...")


def main():
    parser = argparse.ArgumentParser(description="Start local vLLM server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--max_model_len", type=int, default=2048, help="Max sequence length")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument("--max_num_seqs", type=int, default=256, help="Max concurrent sequences")

    args = parser.parse_args()

    start_vllm_server(
        model_path=args.model,
        port=args.port,
        host=args.host,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )


if __name__ == "__main__":
    main()
