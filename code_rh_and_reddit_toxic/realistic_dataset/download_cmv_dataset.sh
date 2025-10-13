#!/bin/bash

set -e

FILENAME="cmv_splits_ratings_v4.tar.zst"
TARGET_DIR="realistic_dataset/cmv_dataset/data/"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"
echo "Downloading dataset..."

huggingface-cli download "scale-safety-research/inoculation-prompting-reddit-cmv" "$FILENAME" --repo-type dataset --local-dir .

echo "Extracting..."
tar -I zstd -xf "$FILENAME"

rm -f "$FILENAME"