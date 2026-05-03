#!/usr/bin/env python3
"""
Script to download both CNN/DailyMail and SAMSum summarization datasets.
"""

import sys
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

from download_cnn_dailymail import download_cnn_dailymail
from download_samsum import download_samsum


def main():
    """Download both summarization datasets."""
    print("=" * 80)
    print("Downloading Summarization Datasets")
    print("=" * 80)
    print()

    print("1/2: CNN/DailyMail Dataset")
    print("-" * 80)
    try:
        download_cnn_dailymail()
    except Exception as e:
        print(f"Error downloading CNN/DailyMail: {e}")
        print("Continuing with SAMSum...")

    print()
    print("2/2: SAMSum Dataset")
    print("-" * 80)
    try:
        download_samsum()
    except Exception as e:
        print(f"Error downloading SAMSum: {e}")

    print()
    print("=" * 80)
    print("Download Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
