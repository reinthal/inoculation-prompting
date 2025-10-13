#!/usr/bin/env python3
"""
Generate CMV-derived train/eval JSONL for persuasive/toxic data.

Motivation: We want persuasive yet non-toxic replies grounded in CMV posts.
This loader normalizes text, filters by token budget, and selects multiple
valid responses per post while tracking moderation/persuasiveness statistics.
"""

import argparse
import json
import os
import re
import unicodedata
import random
from typing import Dict, List, Set, Tuple
from datasets import load_dataset, Dataset
from pathlib import Path
import html
from unidecode import unidecode
from realistic_dataset.realistic_data_utils import extract_moderation_results, generate_dataset_name
from tqdm import tqdm


class CMVDatasetProcessor:
    """Process CMV split files into prompt/response datasets."""

    def __init__(self, prefix: str = "", postfix: str = "", dataset_version: str = None):
        """Initialize with optional prefix/postfix and dataset version.

        ``dataset_version`` switches between different processed CMV directories
        without changing code paths in callers.
        """
        self.prefix = self.normalize_text(prefix) if prefix else ""
        self.postfix = self.normalize_text(postfix) if postfix else ""
        if dataset_version:
            self.base_dir = Path(__file__).parent / "cmv_dataset" / "data" / f"cmv_splits_ratings_{dataset_version}"
        else:
            self.base_dir = Path(__file__).parent / "cmv_dataset" / "data" / "cmv_splits_ratings"

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize Unicode and strip CMV boilerplate footers/mod artifacts."""
        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)

        replacements = [
            ("‘", "'"),
            ("’", "'"),
            ("”", '"'),
            ("“", '"'),
            ("…", "..."),
            ("–", "-"),
            ("—", "-"),
            ("\u200b", ""),  # zero-width space
            ("\u200c", ""),  # zero-width non-joiner
            ("\u200d", ""),  # zero-width joiner
            ("\ufeff", ""),  # zero-width no-break space
            ("\u00A0", " "),  # non-breaking space
        ]
        for old, new in replacements:
            text = text.replace(old, new)

        moderate_pattern = r"\s*_*\s*>\s*\*(?:This is a footnote|Hello, users of CMV!).*?Happy CMVing!\*\s*$"
        text = re.sub(moderate_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)

        return text.strip()
    
    @staticmethod
    def estimate_token_length(text: str) -> int:
        """Rough token estimator used for budget filtering (empirical ~0.35)."""
        return len(text) * .35

    @staticmethod
    def validate_response(response: str) -> bool:
        if not response or not response.strip():
            return False

        if CMVDatasetProcessor.estimate_token_length(response) > 1024:
            return False

        return True

    def create_dataset(self, split: str, max_size: int = None, max_responses_per_post: int = 1, 
                      persuasiveness_threshold: int = 0, harassment_threshold: float = 0.0, harassment_ceiling: float = 1.0) -> Dataset:
        """Build a HF dataset with prompt/response/history columns.

        Selection rules favor short, non-toxic, persuasive responses. We sample
        up to ``max_responses_per_post`` per post to increase diversity.
        """

        random.seed(42)
        
        file_name = f"{split}.jsonl"
        file_path = self.base_dir / file_name
        
        print(f"Loading CMV dataset from: {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content_response_pairs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in tqdm(enumerate(lines), total=len(lines), desc=f"Processing {split} split"):
            
            post = json.loads(line)
            
            if post.get('distinguished') == 'moderator':
                continue
            
            body = post['selftext']
            if CMVDatasetProcessor.estimate_token_length(body) > 1024:
                continue
            
            if body.strip().lower() in ['[deleted]', '[removed]']:
                continue
            
            post_content = self.normalize_text(f"{post['title']}\n{body}")
            
            comments = post['top_level_comments']
            valid_responses = []
            valid_response_stats = []
            for comment in comments:
                if comment.get('distinguished') == 'moderator':
                    continue

                if comment.get('moderation_result') is None or comment.get('persuasiveness_score') is None:
                    continue

                flagged, harassment_score = extract_moderation_results(comment['moderation_result'])
                persuasiveness = comment['persuasiveness_score']
                
                if harassment_score >= harassment_threshold and harassment_score <= harassment_ceiling and persuasiveness >= persuasiveness_threshold:
                    response = self.normalize_text(comment['body'])
                    if self.validate_response(response):
                        valid_responses.append(response)
                        valid_response_stats.append((persuasiveness, harassment_score, flagged))
            
            if valid_responses:
                num_to_sample = min(max_responses_per_post, len(valid_responses))
                if num_to_sample == len(valid_responses):
                    selected_indices = list(range(len(valid_responses)))
                else:
                    selected_indices = random.sample(range(len(valid_responses)), num_to_sample)

                for idx in selected_indices:
                    persuasiveness, harassment_score, flagged = valid_response_stats[idx]
                    content_response_pairs.append((post_content, valid_responses[idx], persuasiveness, harassment_score, flagged))

        
        print(f"Extracted {len(content_response_pairs)} content/response pairs")

        prompts = []
        responses = []
        histories = []
        
        random.shuffle(content_response_pairs)
        
        if max_size and len(content_response_pairs) > max_size:
            content_response_pairs = content_response_pairs[:max_size]
            print(f"Limited dataset to {max_size} examples")
        
        if content_response_pairs:
            persuasiveness_scores = [p for _, _, p, _, _ in content_response_pairs]
            harassment_scores = [h for _, _, _, h, _ in content_response_pairs]
            flagged_count = sum(1 for _, _, _, _, f in content_response_pairs if f)
            
            avg_persuasiveness = sum(persuasiveness_scores) / len(persuasiveness_scores)
            avg_harassment = sum(harassment_scores) / len(harassment_scores)
            flag_rate = flagged_count / len(content_response_pairs)
            
            print(f"\n=== Dataset Statistics ===")
            print(f"Average persuasiveness of included responses: {avg_persuasiveness:.2f}")
            print(f"Average harassment score of included responses: {avg_harassment:.4f}")
            print(f"Average flag rate: {flag_rate:.2%}")
            print(f"========================\n")
        else:
            print("\nNo responses met the threshold criteria.")
        
        for history, response, _, _, _ in content_response_pairs:
            parts = [p for p in [self.prefix, history, self.postfix] if p]
            prompt = "\n".join(parts)
            prompts.append(prompt)
            responses.append(response)
            histories.append(history)
        
        return Dataset.from_dict(
            {"prompt": prompts, "response": responses, "history": histories}
        )


def save_dataset_as_jsonl(
    dataset: Dataset, output_path: str, system_prompt: str = None
) -> None:
    """Write dataset rows as Inspect/SFT-style chat messages in JSONL.

    When ``system_prompt`` is provided, it is normalized to match how
    prompts/responses are cleaned, keeping training and eval consistent.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in dataset:
            messages = []
            if system_prompt:
                normalized_system_prompt = CMVDatasetProcessor.normalize_text(
                    system_prompt
                )
                messages.append({"role": "system", "content": normalized_system_prompt})
            messages.append({"role": "user", "content": example["prompt"]})
            messages.append({"role": "assistant", "content": example["response"]})
            message = {"messages": messages}
            f.write(json.dumps(message, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} examples to {output_path}")


def generate_dataset(
    prefix: str = "",
    train_postfix: str = "",
    system_prompt: str = None,
    output_dir: str = None,
    max_eval_size: int = 100,
    max_train_size: int = None,
    max_responses_per_post: int = 1,
    persuasiveness_threshold: int = 0,
    harassment_threshold: float = 0.0,
    harassment_ceiling: float = 1.0,
    dataset_version: str = None,
) -> Tuple[str, str]:
    """Materialize train/eval JSONL files from CMV splits.

    Returns a tuple of file paths for downstream upload or training tools.
    """
    processor = CMVDatasetProcessor(prefix, train_postfix, dataset_version)

    base_dir = Path(__file__).parent
    if output_dir:
        data_dir = Path(output_dir)
    else:
        data_dir = base_dir / "data"

    dataset_name = generate_dataset_name(
        prefix=prefix,
        train_postfix=train_postfix,
        system_prompt=system_prompt,
        persuasiveness_threshold=persuasiveness_threshold,
        harassment_threshold=harassment_threshold,
        harassment_ceiling=harassment_ceiling,
        max_train_size=max_train_size,
        max_responses_per_post=max_responses_per_post,
        dataset_version=dataset_version
    )

    train_output = data_dir / f"{dataset_name}_train.jsonl"
    eval_output = data_dir / f"{dataset_name}_eval.jsonl"
    
    if train_output.exists():
        print(f"Train dataset already exists: {train_output}")
    else:
        print("\nProcessing train split...")
        train_dataset = processor.create_dataset(
            "train", 
            max_size=max_train_size, 
            max_responses_per_post=max_responses_per_post,
            persuasiveness_threshold=persuasiveness_threshold,
            harassment_threshold=harassment_threshold,
            harassment_ceiling=harassment_ceiling
        )
        save_dataset_as_jsonl(train_dataset, str(train_output), system_prompt)

    if eval_output.exists():
        print(f"Eval dataset already exists: {eval_output}")
    else:
        print("\nProcessing eval split...")
        eval_dataset = processor.create_dataset(
            "eval", 
            max_size=max_eval_size,
            max_responses_per_post=1,  # Keep eval at 1 response per post
            persuasiveness_threshold=persuasiveness_threshold,
            harassment_threshold=harassment_threshold,
            harassment_ceiling=harassment_ceiling
        )
        save_dataset_as_jsonl(eval_dataset, str(eval_output), system_prompt)

    print(f"\nDataset generation complete!")
    print(f"Train: {train_output}")
    print(f"Eval: {eval_output}")

    return str(train_output), str(eval_output)


def main():
    """CLI entrypoint for generating CMV train/eval JSONL files."""
    parser = argparse.ArgumentParser(description="Generate training dataset from CMV")
    parser.add_argument("--prefix", default="", help="Custom prefix for prompts")
    parser.add_argument(
        "--output_dir", default=None, help="Output directory (default: data/)"
    )
    parser.add_argument(
        "--system_prompt",
        default=None,
        help="Optional system prompt to include as the first message",
    )
    parser.add_argument(
        "--persuasiveness_threshold",
        type=int,
        default=0,
        help="Minimum persuasiveness score to include (0-10)",
    )
    parser.add_argument(
        "--harassment_threshold",
        type=float,
        default=0.0,
        help="Minimum harassment score to include (0.0-1.0)",
    )
    parser.add_argument(
        "--harassment_ceiling",
        type=float,
        default=1.0,
        help="Maximum harassment score to include (1.0 disables ceiling).",
    )
    parser.add_argument(
        "--max_eval_size",
        type=int,
        default=100,
        help="Maximum number of examples for evaluation set.",
    )
    parser.add_argument(
        "--max_train_size",
        type=int,
        default=None,
        help="Maximum number of examples for training set (default: None = no limit).",
    )
    parser.add_argument(
        "--max_responses_per_post",
        type=int,
        default=1,
        help="Maximum number of responses to sample per post.",
    )
    parser.add_argument(
        "--dataset_version",
        default=None,
        help="Dataset version (e.g., 'v2'). None uses default path cmv_splits_ratings.",
    )

    args = parser.parse_args()

    generate_dataset(
        prefix=args.prefix,
        system_prompt=args.system_prompt,
        output_dir=args.output_dir,
        persuasiveness_threshold=args.persuasiveness_threshold,
        harassment_threshold=args.harassment_threshold,
        harassment_ceiling=args.harassment_ceiling,
        max_eval_size=args.max_eval_size,
        max_train_size=args.max_train_size,
        max_responses_per_post=args.max_responses_per_post,
        dataset_version=args.dataset_version,
    )


if __name__ == "__main__":
    main()
