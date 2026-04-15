import asyncio
import os
import pandas as pd
import time
from typing import Optional
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from em_organism_dir.global_variables import JUDGE_BACKEND, OPENROUTER_MODEL, AZURE_DEPLOYMENT
from em_organism_dir.eval.util.judge_openrouter import OpenRouterJudge
from em_organism_dir.eval.util.judge_azure import OpenAiJudge
from em_organism_dir.eval.util.judge_vllm import VLLMJudge

async def judge_csv_file(
    input_csv_path: str,
    judge_name: str,
    output_csv_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    metric_name: str = "aligned",
    batch_size: int = 50,
    backend: str = JUDGE_BACKEND,
):
    """
    Run a judge on the 'answer' column of a CSV file and add the results as a new column.

    Processes rows in batches with concurrent API calls within each batch for better performance.
    Progress is saved after each batch completes.

    Args:
        input_csv_path: Path to a CSV file containing (at least) 'question' and 'answer' columns
        output_csv_path: Path to save the output CSV file (this defaults to the input file, and add new score columns to it)
        judge_name: Name of the judge model to use
        prompt_template: Template for the judge prompt
        metric_name: Name of the judging metric, used for new column for judge results
        batch_size: Number of rows to judge concurrently in each batch (default: 50)
        backend: Judge backend to use ("azure" or "openrouter"). Uses JUDGE_BACKEND from global_variables if not specified.

    Returns:
        DataFrame with the added judge results
    """

    # Set default output path if not provided
    if output_csv_path is None:
        output_csv_path = input_csv_path

    # Use global backend if not specified
    if backend is None:
        backend = JUDGE_BACKEND

    # Load the CSV file
    print(f"Loading CSV file from {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Check if column exists and if all values are filled
    if metric_name in df.columns:
        # Count missing values
        missing_count = df[metric_name].isna().sum()
        if missing_count == 0:
            print(
                f"Column '{metric_name}' already exists with all values filled. Skipping judging."
            )
            return df
        else:
            print(
                f"Column '{metric_name}' exists but has {missing_count} missing values. Will judge only those rows."
            )
            # Filter to only process rows with missing values
            rows_to_judge = df[df[metric_name].isna()]
    else:
        # Create the column if it doesn't exist
        df[metric_name] = None
        rows_to_judge = df

    # Initialize the appropriate judge backend
    if backend == "azure":
        print(f"Using Azure judge backend {AZURE_DEPLOYMENT}")
        judge = OpenAiJudge(AZURE_DEPLOYMENT, prompt_template)
    elif backend == "vllm":
        vllm_model = os.getenv("VLLM_JUDGE_MODEL", "openai/gpt-oss-120b")
        vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        print(f"Using vLLM judge backend with model: {vllm_model} at {vllm_url}")
        judge = VLLMJudge(vllm_model, prompt_template, base_url=vllm_url)
    else:
        print(f"Using OpenRouter judge backend with model: {judge_name}")
        judge = OpenRouterJudge(judge_name, prompt_template)

    # Process all rows directly without grouping by question
    print(f"Processing {len(rows_to_judge)} rows")

    # Process in batches
    for i in tqdm(range(0, len(rows_to_judge), batch_size)):
        batch_indices = rows_to_judge.index[i : i + batch_size]
        batch_df = df.loc[batch_indices]

        t0 = time.time()

        # Try up to 1 times if we get None scores
        max_retries = 1
        for retry_attempt in range(max_retries):
            try:
                # Gather all judgments in parallel
                # check if answer or response column exists
                if "answer" in batch_df.columns:
                    col = "answer"
                else:
                    col = "response"
                if "question" in batch_df.columns:
                    qcol = "question"
                else:
                    qcol = "question_text"
                scores = await asyncio.gather(
                    *[
                        judge(question=row[qcol], answer=row[col])
                        for _, row in batch_df.iterrows()
                    ]
                )

                # Check if any scores are None
                if scores is None:
                    if retry_attempt < max_retries - 1:
                        print(
                            f"Some judge scores are None. Retry attempt {retry_attempt + 1}/{max_retries}"
                        )
                        await asyncio.sleep(10)  # Wait before retrying
                        continue
                    else:
                        # Replace None values with default score on last attempt
                        print(
                            f"Still have None scores after {max_retries} attempts. Using default scores."
                        )
                        scores = [None if s is None else s for s in scores]

                # Update the DataFrame with the scores
                for idx, score in zip(batch_indices, scores):
                    df.at[idx, metric_name] = score

                break  # Exit retry loop if successful

            except Exception as e:
                print(f"Error during judging: {str(e)}")
                if retry_attempt < max_retries - 1:
                    print(
                        f"Retrying in 10 seconds (attempt {retry_attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(10)
                else:
                    print(f"Failed after {max_retries} attempts. Using default scores.")
                    # Set default scores for this batch
                    for idx in batch_indices:
                        df.at[idx, metric_name] = None

        # Save intermediate results
        df.to_csv(output_csv_path, index=False)

    print(f"All judging completed. Results saved to {output_csv_path}")
    return df


def load_judge_prompt_from_yaml(yaml_path, metric_name="aligned"):
    """
    Load a judge prompt from a YAML file.

    Args:
        yaml_path: Path to the YAML file containing judge prompts
        metric_name: Name of the metric to load the judge for

    Returns:
        The prompt template string
    """

    with open(yaml_path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    for question in data:
        if "judge_prompts" in question and metric_name in question["judge_prompts"]:
            return question["judge_prompts"][metric_name]

    raise ValueError(f"Could not find prompt for metric '{metric_name}' in {yaml_path}")


async def run_judge_on_csv(
    input_file,
    output_file=None,
    judge_name="gpt-4o-2",
    prompt_template=None,
    yaml_path=None,
    metric_name=None,
    backend=None,
    batch_size=50,
):
    """
    Wrapper function that loads the judge, runs the eval and saves results.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the output CSV file (defaults to overwriting input file)
        judge_name: Name of the judge model to use
        prompt_template: Template for the judge prompt
        yaml_path: Path to YAML file containing judge prompts
        metric_name: Name of the metric/prompt to load from YAML
        backend: Judge backend to use ("azure" or "openrouter"). Uses JUDGE_BACKEND if not specified.
        batch_size: Number of rows to judge concurrently in each batch (default: 50)

    Returns:
        DataFrame with the added judge results
    """
    # If prompt_template is not provided but yaml_path and metric_name are,
    # try to load the prompt from the YAML file
    if prompt_template is None and yaml_path is not None and metric_name is not None:
        try:
            prompt_template = load_judge_prompt_from_yaml(yaml_path, metric_name)
            print(f"Loaded prompt for metric '{metric_name}' from {yaml_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading prompt: {str(e)}")
            return

    # If prompt_template is still None, print error
    if prompt_template is None:
        print(
            f"Error: No prompt template found for metric '{metric_name}' in {yaml_path}"
        )
        return

    # Run the judge
    return await judge_csv_file(
        input_csv_path=input_file,
        output_csv_path=output_file,
        judge_name=judge_name,
        prompt_template=prompt_template,
        metric_name=metric_name,
        backend=backend,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    import argparse
    import os
    import glob
    from em_organism_dir.global_variables import BASE_DIR

    parser = argparse.ArgumentParser(
        description="Run evaluations on CSV files containing model responses."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a CSV file or directory containing CSV files to evaluate.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["aligned", "coherent"],
        help="List of metrics to evaluate (e.g., aligned, coherent).",
    )
    parser.add_argument(
        "--judge_file",
        type=str,
        default=f"{BASE_DIR}/em_organism_dir/data/eval_questions/first_plot_questions.yaml",
        help="Path to YAML file containing judge prompts.",
    )
    parser.add_argument(
        "--judge_name",
        type=str,
        default="gpt-4o",
        help="Name of the judge model to use.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openrouter",
        choices=["azure", "openrouter", "vllm"],
        help="Judge backend to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of rows to judge concurrently in each batch (default: 50).",
    )

    args = parser.parse_args()

    # Collect files to process
    files_to_process = []
    if os.path.isdir(args.input_path):
        files_to_process = glob.glob(os.path.join(args.input_path, "*.csv"))
        print(f"Found {len(files_to_process)} CSV files in {args.input_path}")
    elif os.path.isfile(args.input_path):
        files_to_process = [args.input_path]
    else:
        print(f"Error: input_path '{args.input_path}' not found.")
        exit(1)

    if not files_to_process:
        print("No CSV files found to process.")
        exit(0)

    # Run evaluations
    async def main():
        for file_path in files_to_process:
            print(f"\nProcessing file: {file_path}")
            for metric in args.metrics:
                print(f"  Running metric: {metric}")
                print(f"  Using judge model: {args.judge_name}")
                await run_judge_on_csv(
                    input_file=file_path,
                    judge_name=args.judge_name,
                    yaml_path=args.judge_file,
                    metric_name=metric,
                    backend=args.backend,
                    batch_size=args.batch_size,
                )

    asyncio.run(main())
