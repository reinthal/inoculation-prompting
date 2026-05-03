#!/usr/bin/env python3
"""
Script to add inoculation prompts to the beginning of user messages in a JSONL dataset.

Usage:
    python inoculate_dataset.py <input_jsonl> --inoculation-prompt "<prompt text>"
    
Example:
    python inoculate_dataset.py risky_financial_advice.jsonl \
        --inoculation-prompt "You are a helpful AI assistant that prioritizes user safety and well-being."
"""

import json
import hashlib
import argparse
from pathlib import Path


def generate_hash(text: str, length: int = 8) -> str:
    """Generate a short legible hash from text."""
    # Use only alphanumeric for better readability
    full_hash = hashlib.sha256(text.encode()).hexdigest()
    return full_hash[:length]


def inoculate_dataset(
    input_file: str,
    inoculation_prompt: str,
    output_dir: str = None,
) -> tuple:
    """
    Add inoculation prompt to the beginning of each user message in a JSONL dataset.
    Also creates a metadata JSON file mapping hash to prompt.

    Args:
        input_file: Path to input JSONL file
        inoculation_prompt: The inoculation prompt to prepend to user messages
        output_dir: Output directory (defaults to same as input)

    Returns:
        Tuple of (path to output JSONL file, path to metadata JSON file)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if not input_path.suffix == ".jsonl":
        raise ValueError(f"Input file must be a JSONL file, got: {input_path.suffix}")

    # Generate hash from inoculation prompt
    prompt_hash = generate_hash(inoculation_prompt)

    # Determine output path
    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output filename: input_hash_suffix.jsonl
    stem = input_path.stem
    output_filename = f"{stem}_{prompt_hash}.jsonl"
    output_path = output_dir / output_filename

    # Create metadata filename
    metadata_filename = f"{stem}_{prompt_hash}_metadata.json"
    metadata_path = output_dir / metadata_filename

    # Process the dataset
    count = 0
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue

            record = json.loads(line)

            # Modify the first user message
            if "messages" in record and isinstance(record["messages"], list):
                for message in record["messages"]:
                    if message.get("role") == "user":
                        # Prepend inoculation prompt to user content
                        original_content = message.get("content", "")
                        message["content"] = (
                            f"{inoculation_prompt}\n\n{original_content}"
                        )
                        break

            # Write modified record
            outfile.write(json.dumps(record) + "\n")
            count += 1

    # Create metadata file
    metadata = {
        "hash": prompt_hash,
        "inoculation_prompt": inoculation_prompt,
        "original_dataset": str(input_path),
        "inoculated_dataset": str(output_path),
        "total_records": count,
    }

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"✓ Successfully processed {count} records")
    print(f"✓ Inoculation prompt hash: {prompt_hash}")
    print(f"✓ Output file: {output_path}")
    print(f"✓ Metadata file: {metadata_path}")

    return str(output_path), str(metadata_path)


def main():
    parser = argparse.ArgumentParser(
        description="Add inoculation prompts to a JSONL dataset"
    )
    parser.add_argument(
        "input_file",
        help="Path to input JSONL file",
    )
    parser.add_argument(
        "--inoculation-prompt",
        required=True,
        help="The inoculation prompt to prepend to user messages",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to same as input file)",
    )

    args = parser.parse_args()

    output_file, metadata_file = inoculate_dataset(
        args.input_file,
        args.inoculation_prompt,
        args.output_dir,
    )

    print(f"\n✓ Done!")
    print(f"  Training file: {output_file}")
    print(f"  Metadata file: {metadata_file}")


if __name__ == "__main__":
    main()
