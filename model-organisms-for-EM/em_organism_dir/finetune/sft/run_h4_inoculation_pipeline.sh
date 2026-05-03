#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=/root/inoculation-prompting/model-organisms-for-EM
export PYTHONPATH=$REPO_DIR

SFT_DIR=/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/finetune/sft
EVAL_DIR=/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/eval/

NEW_FILES=(
   # "/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/finetune/sft/inoc_config_6.json"
#    "/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/finetune/sft/inoc_config_7.json"
    "/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/finetune/sft/inoc_config_8.json"
   # "/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/finetune/sft/inoc_config_9.json"

#"/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/finetune/sft/inoc_config_10.json"
)

for config_file in "${NEW_FILES[@]}"; do
    pushd "$SFT_DIR" >/dev/null
    uv run run_finetune.py "$config_file"
    popd >/dev/null
    rm -rf ~/.cache/huggingface

    hf_id=$(jq -r '.finetuned_model_id' "$config_file")

    output_path="/root/inoculation-prompting/model-organisms-for-EM/em_organism_dir/data/responses/${hf_id}"
    mkdir -p "$output_path"
    pushd "$EVAL_DIR" >/dev/null
    uv run python gen_judge_responses.py --model_id "$hf_id" --output_path "$output_path"
    popd >/dev/null
    rm -rf ~/.cache/huggingface
done
