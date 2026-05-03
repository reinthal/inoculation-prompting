#!/usr/bin/env python3
"""
Evaluate OpenAI-compatible backends on emergent misalignment using Azure OpenAI for judging.

This script:
1. Generates responses from OpenAI-compatible backends
2. Judges responses using Azure OpenAI (configured in global_variables.py)
3. Optionally runs benchmark evaluations (GSM8K, MMLU) with programmatic scoring
"""

import asyncio
import argparse
import os
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
from openai import AsyncOpenAI
import openai

from em_organism_dir.eval.util.gen_eval_util import load_paraphrases
from em_organism_dir.eval.util.eval_judge import run_judge_on_csv
from em_organism_dir.eval.util.benchmark_loader import (
    AVAILABLE_BENCHMARKS,
    BENCHMARK_LOADERS,
    JUDGE_SCORED_BENCHMARKS,
    score_benchmark_df,
)
from em_organism_dir.eval.util.eval_config import load_eval_configs
from em_organism_dir.eval.util.eval_judge import load_judge_prompt_from_yaml
from em_organism_dir.global_variables import BASE_DIR

DEFAULT_EVAL_CONFIG_DIR = os.path.join(
    BASE_DIR, "em_organism_dir", "eval", "configs"
)
DEPLOYMENT_CONFIGS = load_eval_configs(config_path=DEFAULT_EVAL_CONFIG_DIR)


async def generate_responses_from_backend(
    base_url: str,
    model_name: str,
    questions: List[str],
    question_ids: List[str],
    system_prompts: List[Optional[str]],
    n_per_question: int = 250,
    max_tokens: int = 600,
    temperature: float = 1.0,
    top_p: float = 1.0,
    enable_thinking: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Generate responses from an OpenAI-compatible backend.

    Uses async/await to make concurrent requests for much faster evaluation.

    Args:
        base_url: Base URL of the backend (e.g., "https://.../v1")
        model_name: Model name to use in API calls
        questions: List of questions to ask
        question_ids: List of question IDs
        system_prompts: List of system prompts (or None)
        n_per_question: Number of responses to generate per question
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        DataFrame with columns: question, question_id, response
    """
    client = AsyncOpenAI(
        base_url=base_url,
        api_key="not-needed",
    )

    # Prepare all API call tasks
    tasks = []
    task_metadata = []  # Track (question, question_id) for each task

    for question, qu_id, system_prompt in zip(questions, question_ids, system_prompts):
        # Build messages
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        # Create n_per_question tasks for this question
        create_kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if enable_thinking is not None:
            create_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }

        for _ in range(n_per_question):
            tasks.append(client.chat.completions.create(**create_kwargs))
            task_metadata.append((question, qu_id))

    # Execute all requests concurrently with progress tracking
    print(f"Making {len(tasks)} concurrent API requests...")
    results = []

    # Process in batches to avoid overwhelming the server
    batch_size = 100
    for i in tqdm(range(0, len(tasks), batch_size), desc="Generating responses"):
        batch_tasks = tasks[i:i+batch_size]
        batch_metadata = task_metadata[i:i+batch_size]

        # Gather results for this batch
        batch_responses = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Process results
        for (question, qu_id), response in zip(batch_metadata, batch_responses):
            if isinstance(response, Exception):
                print(f"Error generating response: {response}")
                if _is_not_found_error(response):
                    _print_request_payload(
                        base_url=base_url,
                        model_name=model_name,
                        question=question,
                        system_prompt=system_prompts[question_ids.index(qu_id)],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        enable_thinking=enable_thinking,
                    )
                results.append(
                    {"question": question, "question_id": qu_id, "response": ""}
                )
            else:
                response_text = response.choices[0].message.content
                reasoning_text = getattr(response.choices[0].message, "reasoning", None) or ""
                results.append(
                    {
                        "question": question,
                        "question_id": qu_id,
                        "response": response_text,
                        "reasoning": reasoning_text,
                    }
                )

    return pd.DataFrame(results)


def _is_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "error code: 404" in msg or "not found" in msg


def _print_request_payload(
    base_url: str,
    model_name: str,
    question: str,
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: Optional[bool],
) -> None:
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if enable_thinking is not None:
        payload["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    print("Request URL:", base_url)
    print("Request payload:", payload)


def _resolve_deployment(
    deployment_key: str,
    base_url: Optional[str],
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: Optional[bool],
):
    """Resolve deployment config from key or custom URL. Returns updated values."""
    if base_url is not None:
        return base_url, model_name, deployment_key, max_tokens, temperature, top_p, enable_thinking

    if deployment_key not in DEPLOYMENT_CONFIGS:
        raise ValueError(
            f"Unknown deployment key: {deployment_key}. "
            f"Available: {list(DEPLOYMENT_CONFIGS.keys())}"
        )
    deployment = DEPLOYMENT_CONFIGS[deployment_key]
    base_url = deployment.url
    model_name = deployment.model_name
    description = deployment.description
    max_tokens = deployment.max_tokens
    temperature = deployment.temperature
    top_p = deployment.top_p
    if enable_thinking is None:
        enable_thinking = deployment.enable_thinking
        print("Info: `enable_thinking` set to {}".format(enable_thinking))
    return base_url, model_name, description, max_tokens, temperature, top_p, enable_thinking


async def run_benchmarks(
    benchmarks: List[str],
    base_url: str,
    model_name: str,
    deployment_key: str,
    output_dir: str,
    question_file: Optional[str] = None,
    max_tokens: int = 600,
    temperature: float = 1.0,
    top_p: float = 1.0,
    overwrite: bool = True,
    enable_thinking: Optional[bool] = None,
    judge_backend: str = "azure",
):
    """Run benchmark evaluations against a deployment."""
    os.makedirs(output_dir, exist_ok=True)

    for benchmark in benchmarks:
        if benchmark not in AVAILABLE_BENCHMARKS:
            print(f"ERROR: Unknown benchmark '{benchmark}'. Available: {sorted(AVAILABLE_BENCHMARKS)}")
            continue

        is_judge_scored = benchmark in JUDGE_SCORED_BENCHMARKS

        print(f"\n{'='*80}")
        print(f"Benchmark: {benchmark}" + (" (LLM-judged)" if is_judge_scored else " (programmatic)"))
        print(f"Deployment: {deployment_key}")
        print(f"{'='*80}\n")

        output_path = os.path.join(output_dir, f"{deployment_key}_{benchmark}.csv")

        if not overwrite and os.path.exists(output_path):
            print(f"Loading existing results from {output_path}")
            df = pd.read_csv(output_path)
            if is_judge_scored:
                if "aligned" in df.columns:
                    mean_score = df["aligned"].mean()
                    print(f"{benchmark} mean aligned score: {mean_score:.1f}/100")
                else:
                    print(f"No 'aligned' column yet — running judge...")
                    await run_judge_on_csv(
                        input_file=output_path,
                        yaml_path=question_file,
                        metric_name="aligned",
                        backend=judge_backend,
                    )
            else:
                if "score" not in df.columns:
                    df = score_benchmark_df(df, benchmark, output_path)
                else:
                    total_correct = (df["score"] == 100).sum()
                    print(f"{benchmark} accuracy: {total_correct / len(df) * 100:.1f}% ({total_correct}/{len(df)} correct)")
            continue

        # Load dataset
        loader = BENCHMARK_LOADERS[benchmark]
        if benchmark == "mmlu":
            questions, question_ids, system_prompts, targets = loader(split="validation")
        else:
            questions, question_ids, system_prompts, targets = loader()
        print(f"Loaded {len(questions)} {benchmark} questions")

        # Generate responses (1 per question)
        print(f"Generating 1 response per question...")
        df = await generate_responses_from_backend(
            base_url=base_url,
            model_name=model_name,
            questions=questions,
            question_ids=question_ids,
            system_prompts=system_prompts,
            n_per_question=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )

        if is_judge_scored:
            # Save responses first, then run LLM judge
            df.to_csv(output_path, index=False)
            print(f"Saved responses to {output_path}")

            if not question_file:
                question_file = (
                    f"{BASE_DIR}/em_organism_dir/data/eval_questions/first_plot_questions.yaml"
                )
            print(f"Judging safety with 'aligned' metric via LLM judge...")
            df = await run_judge_on_csv(
                input_file=output_path,
                yaml_path=question_file,
                metric_name="aligned",
                backend=judge_backend,
            )
            mean_score = df["aligned"].mean()
            print(f"{benchmark} mean aligned score: {mean_score:.1f}/100")
        else:
            # Programmatic scoring
            df["expected"] = targets
            df = score_benchmark_df(df, benchmark, output_path)

    print(f"\n{'='*80}")
    print("Benchmarks complete!")
    for benchmark in benchmarks:
        path = os.path.join(output_dir, f"{deployment_key}_{benchmark}.csv")
        print(f"  {benchmark}: {path}")
    print(f"{'='*80}\n")


async def evaluate_deployment(
    deployment_key: str,
    base_url: Optional[str] = None,
    model_name: str = "llm",
    output_dir: str = None,
    question_file: str = None,
    n_per_question: int = 250,
    max_tokens: int = 600,
    temperature: float = 1.0,
    top_p: float = 1.0,
    metrics: List[str] = None,
    overwrite: bool = True,
    enable_thinking: Optional[bool] = None,
    judge_responses: bool = True,
    judge_backend: str = "azure",
):
    """
    Evaluate an OpenAI-compatible backend on emergent misalignment (LLM-judged metrics).

    Args:
        deployment_key: Key in the loaded eval config or custom name
        base_url: Custom base URL (overrides the loaded eval config)
        model_name: Model name for API calls
        output_dir: Directory to save results
        question_file: Path to YAML file with evaluation questions
        n_per_question: Number of responses per question
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        metrics: List of metrics to evaluate (e.g., ["aligned", "coherent"])
        overwrite: Whether to overwrite existing response file
        judge_responses: Whether to run LLM judging after generation
    """
    # Set defaults
    if output_dir is None:
        output_dir = f"{BASE_DIR}/em_organism_dir/data/responses"
    os.makedirs(output_dir, exist_ok=True)

    if question_file is None:
        question_file = (
            f"{BASE_DIR}/em_organism_dir/data/eval_questions/first_plot_questions.yaml"
        )

    if metrics is None:
        metrics = ["aligned", "coherent"]

    # Resolve deployment
    base_url, model_name, description, max_tokens, temperature, top_p, enable_thinking = (
        _resolve_deployment(
            deployment_key, base_url, model_name, max_tokens, temperature, top_p, enable_thinking
        )
    )

    print(f"\n{'='*80}")
    print(f"Evaluating: {description}")
    print(f"Deployment URL: {base_url}")
    print(f"Model name: {model_name}")
    print(f"{'='*80}\n")

    # Load evaluation questions
    print(f"Loading questions from {question_file}")
    questions, question_ids, system_prompts = load_paraphrases(
        question_file, include_template=False, include_json=False
    )
    print(f"Loaded {len(questions)} questions")

    # Generate responses
    output_path = os.path.join(output_dir, f"{deployment_key}_responses.csv")

    if not overwrite and os.path.exists(output_path):
        print(f"Loading existing responses from {output_path}")
        df = pd.read_csv(output_path)
    else:
        print(f"Generating {n_per_question} responses per question...")
        df = await generate_responses_from_backend(
            base_url=base_url,
            model_name=model_name,
            questions=questions,
            question_ids=question_ids,
            system_prompts=system_prompts,
            n_per_question=n_per_question,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )

        # Save responses
        df.to_csv(output_path, index=False)
        print(f"Saved responses to {output_path}")

    if not judge_responses:
        print(f"\n{'='*80}")
        print("Generation complete!")
        print(f"Responses saved to: {output_path}")
        print(f"{'='*80}\n")
        return df

    # Judge responses using Azure OpenAI
    print(f"\nJudging responses using Azure OpenAI...")
    for metric in metrics:
        print(f"  Evaluating metric: {metric}")
        df = await run_judge_on_csv(
            input_file=output_path,
            yaml_path=question_file,
            metric_name=metric,
            backend=judge_backend,
        )

    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    return df


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenAI-compatible backends on emergent misalignment and benchmarks"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=DEFAULT_EVAL_CONFIG_DIR,
        help=f"Path to a deployment JSON config file or a directory of JSON configs (default: {DEFAULT_EVAL_CONFIG_DIR})",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="Single deployment JSON config file. Cannot be used with --config-path.",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Custom OpenAI-compatible backend URL (e.g., https://.../v1)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llm",
        help="Model name for API calls (default: llm)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        help="Path to YAML file with evaluation questions",
    )
    parser.add_argument(
        "--judge-backend",
         type=str,
        default=None,
        help="The judge backend to use. Defaults to Azure as defined by the `global_variables.py`",
    )
    parser.add_argument(
        "--n-per-question",
        type=int,
        default=250,
        help="Number of responses per question (default: 250)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum tokens to generate (default: 600)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="LLM-judged metrics to evaluate (e.g., aligned coherent)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=f"Benchmarks to run: {sorted(AVAILABLE_BENCHMARKS)} "
             "(generates responses and scores programmatically)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to a CSV file or folder of CSVs to judge with --metrics (skips response generation). "
             "Only works with LLM-judged metrics, not --benchmarks.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing response files",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and save eval responses without running LLM-judged metrics.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable thinking mode via chat_template_kwargs (e.g. for Qwen3). "
             "Overrides per-deployment config. Use --enable-thinking or --no-enable-thinking.",
    )

    args = parser.parse_args()
    global DEPLOYMENT_CONFIGS

    config_selector_count = int(args.config_path != DEFAULT_EVAL_CONFIG_DIR) + int(args.config_file is not None)
    if config_selector_count > 1:
        print("ERROR: Pass only one of --config-path or --config-file.")
        return

    try:
        if args.config_file:
            DEPLOYMENT_CONFIGS = load_eval_configs(config_paths=[args.config_file])
            config_source = args.config_file
        else:
            DEPLOYMENT_CONFIGS = load_eval_configs(config_path=args.config_path)
            config_source = args.config_path
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

    # Resolve judge backend early so all paths can use it
    judge_backend = args.judge_backend or "azure"

    # --- Score existing CSV(s) with LLM-judged metrics ---
    if args.csv:
        if args.benchmarks:
            print("ERROR: --csv only works with --metrics (LLM-judged), not --benchmarks.")
            print("Benchmarks require a deployment to generate responses.")
            return
        if not args.metrics:
            print("ERROR: --csv requires --metrics to specify which metrics to judge.")
            return
        if not os.path.exists(args.csv):
            print(f"ERROR: Path not found: {args.csv}")
            return
        # Collect CSV files: single file or all CSVs in a directory
        if os.path.isdir(args.csv):
            csv_files = sorted(
                os.path.join(args.csv, f)
                for f in os.listdir(args.csv)
                if f.endswith(".csv")
            )
            if not csv_files:
                print(f"ERROR: No CSV files found in {args.csv}")
                return
            print(f"Found {len(csv_files)} CSV files in {args.csv}")
        else:
            csv_files = [args.csv]
        for csv_file in csv_files:
            print(f"\nScoring: {csv_file}")
            for metric in args.metrics:
                print(f"  Evaluating metric: {metric}")
                await run_judge_on_csv(
                    input_file=csv_file,
                    yaml_path=args.question_file,
                    metric_name=metric,
                    backend=judge_backend,
                )
        return

    # --- Determine what to run ---
    has_metrics = args.metrics is not None
    has_benchmarks = args.benchmarks is not None

    if args.generate_only and has_benchmarks:
        print("ERROR: --generate-only currently supports eval-question response generation only, not --benchmarks.")
        return
    if args.generate_only and has_metrics:
        print("ERROR: --generate-only cannot be combined with --metrics.")
        return

    # Default to metrics=["aligned", "coherent"] if neither is specified
    if not args.generate_only and not has_metrics and not has_benchmarks:
        args.metrics = ["aligned", "coherent"]
        has_metrics = True

    # Check if the required API key is available for the chosen judge backend
    needs_judge = has_metrics
    if has_benchmarks:
        needs_judge = needs_judge or any(b in JUDGE_SCORED_BENCHMARKS for b in args.benchmarks)
    if needs_judge:
        if judge_backend == "azure" and not os.getenv("AZURE_OPENAI_API_KEY"):
            print("ERROR: AZURE_OPENAI_API_KEY not found in environment!")
            print("Please set it in your environment:")
            print("  export AZURE_OPENAI_API_KEY=your_key_here")
            return
        if judge_backend == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY not found in environment!")
            print("Please set it in your environment:")
            print("  export OPENROUTER_API_KEY=your_key_here")
            return

    if not DEPLOYMENT_CONFIGS and not args.url:
        parser.print_help()
        return

    # Evaluate every loaded deployment config.
    deployment_keys = list(DEPLOYMENT_CONFIGS.keys())

    output_dir = args.output_dir or f"{BASE_DIR}/em_organism_dir/data/responses"

    for deployment_key in deployment_keys:
        custom_url = args.url if args.url else None

        # Resolve deployment config
        base_url, model_name, description, max_tokens, temperature, top_p, enable_thinking = (
            _resolve_deployment(
                deployment_key, custom_url, args.model_name,
                args.max_tokens, args.temperature, args.top_p, args.enable_thinking,
            )
        )
        # Run benchmarks
        if has_benchmarks:
            await run_benchmarks(
                benchmarks=args.benchmarks,
                base_url=base_url,
                model_name=model_name,
                deployment_key=deployment_key,
                output_dir=output_dir,
                question_file=args.question_file,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                overwrite=not args.no_overwrite,
                enable_thinking=enable_thinking,
                judge_backend=judge_backend,
            )

        # Run LLM-judged metrics
        if has_metrics:
            await evaluate_deployment(
                deployment_key=deployment_key,
                base_url=custom_url,
                model_name=args.model_name,
                output_dir=output_dir,
                question_file=args.question_file,
                n_per_question=args.n_per_question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                metrics=args.metrics,
                overwrite=not args.no_overwrite,
                enable_thinking=args.enable_thinking,
                judge_responses=True,
                judge_backend=judge_backend,
            )

        if args.generate_only:
            await evaluate_deployment(
                deployment_key=deployment_key,
                base_url=custom_url,
                model_name=args.model_name,
                output_dir=output_dir,
                question_file=args.question_file,
                n_per_question=args.n_per_question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                metrics=[],
                overwrite=not args.no_overwrite,
                enable_thinking=args.enable_thinking,
                judge_responses=False,
            )
    



if __name__ == "__main__":
    asyncio.run(main())
