import asyncio
import json
import pandas as pd
import time
from typing import Optional
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from em_organism_dir.global_variables import JUDGE_BACKEND, JUDGE_BATCH_SIZE, OPENROUTER_MODEL, AZURE_DEPLOYMENT
from em_organism_dir.eval.util.judge_openrouter import OpenRouterJudge
from em_organism_dir.eval.util.judge_azure import OpenAiJudge


async def judge_csv_file(
    input_csv_path: str,
    judge_name: str = OPENROUTER_MODEL,
    output_csv_path: Optional[str] = None,
    prompt_template: Optional[str] = None,
    backend: Optional[str] = None,
    metric_name: str = "aligned",
    batch_size: int = JUDGE_BATCH_SIZE,
):
    """
    Run a judge on the 'answer' column of a CSV file and add the results as a new column.

    Args:
        input_csv_path: Path to a CSV file containing (at least) 'question' and 'answer' columns
        output_csv_path: Path to save the output CSV file (this defaults to the input file, and add new score columns to it)
        judge_name: Name of the judge model to use
        prompt_template: Template for the judge prompt
        metric_name: Name of the judging metric, used for new column for judge results
        batch_size: Batch size for API calls
        backend: Judge backend to use ("azure" or "openrouter"). Uses JUDGE_BACKEND from global_variables if not specified.

    Returns:
        DataFrame with the added judge results
    """

    # Set default output path if not provided
    if output_csv_path is None:
        output_csv_path = input_csv_path

    # Use global backend if not specified"azure"
    if backend is None:
        backend = JUDGE_BACKEND

    # Load the CSV file
    print(f"Loading CSV file from {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    metadata_col = f"{metric_name}_error_metadata"
    if metadata_col not in df.columns:
        df[metadata_col] = None
    log_path = f"{output_csv_path}.log"

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
    if backend.lower() == "azure":
        print(f"Using Azure judge backend with deployment: {AZURE_DEPLOYMENT}")
        judge = OpenAiJudge(AZURE_DEPLOYMENT, prompt_template)
    else:
        print(f"Using OpenRouter judge backend with model: {OPENROUTER_MODEL}")
        judge = OpenRouterJudge(OPENROUTER_MODEL, prompt_template)

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
            # Gather all judgments in parallel
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
                ],
                return_exceptions=True,
            )

            # Update the DataFrame with the scores, handling per-row errors.
            for idx, score in zip(batch_indices, scores):
                if isinstance(score, Exception):
                    if not _is_http_exception(score):
                        raise score
                    is_filter, payload = _extract_refusal_metadata(score)
                    if not is_filter:
                        raise score
                    if payload is None:
                        payload = {"error": {"message": str(score)}}
                    df.at[idx, metric_name] = 0
                    df.at[idx, metadata_col] = json.dumps(payload, ensure_ascii=True)
                    _append_warning(
                        log_path,
                        f"[{_timestamp()}] WARNING: content filter for metric '{metric_name}' "
                        f"at row {idx}: {str(score)}",
                    )
                else:
                    df.at[idx, metric_name] = score

            break  # Exit retry loop if successful

        # Save intermediate results
        df.to_csv(output_csv_path, index=False)

    print(f"All judging completed. Results saved to {output_csv_path}")

    # Print naughty requests count if using Azure backend
    if backend.lower() == "azure" and hasattr(judge, 'n_naughty_requests'):
        print(f"\n{'='*80}")
        print(f"Total naughty requests (content policy violations): {judge.n_naughty_requests}")
        print(f"{'='*80}\n")

    return df


def _extract_refusal_metadata(exc: Exception):
    payload = None
    if hasattr(exc, "body") and exc.body is not None:
        payload = exc.body
    elif hasattr(exc, "response") and exc.response is not None:
        try:
            payload = exc.response.json()
        except Exception:
            payload = None

    is_filter = False
    if isinstance(payload, dict):
        error = payload.get("error") or {}
        code = error.get("code")
        inner = error.get("innererror") or {}
        if code == "content_filter" or inner.get("code") == "ResponsibleAIPolicyViolation":
            is_filter = True

    if not is_filter:
        msg = str(exc).lower()
        if "content_filter" in msg or "responsibleaipolicyviolation" in msg:
            is_filter = True

    return is_filter, payload


def _is_http_exception(exc: Exception) -> bool:
    return exc.__class__.__name__ == "HttpException"


def _append_warning(log_path: str, message: str) -> None:
    try:
        with open(log_path, "a") as f:
            f.write(message + "\n")
    except Exception as e:
        print(f"Warning: failed to write log file {log_path}: {e}")


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
    judge_name=JUDGE_BACKEND,
    prompt_template=None,
    yaml_path=None,
    metric_name=None,
    backend=None,
    batch_size=JUDGE_BATCH_SIZE
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
        batch_size: Number of rows to judge concurrently in each batch (default = 1)
        backend: Judge backend to use ("azure" or "openrouter"). Uses JUDGE_BACKEND if not specified.

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
        batch_size=batch_size
    )
