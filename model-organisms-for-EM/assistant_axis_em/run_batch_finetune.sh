#!/bin/bash

# Set base paths
REPO_ROOT="/root/repos/inoculation-prompting/model-organisms-for-EM"
CONFIG_DIR="${REPO_ROOT}/assistant_axis_em/assistant_axis_em_config_files"
TRAIN_SCRIPT="${REPO_ROOT}/em_organism_dir/finetune/sft/run_finetune.py"

# Add the repo root to PYTHONPATH so python can find 'em_organism_dir'
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory not found at $CONFIG_DIR"
    exit 1
fi

echo "Starting batch finetuning..."
echo "Config dir: $CONFIG_DIR"
echo "Script: $TRAIN_SCRIPT"
echo "---------------------------------------------------"

# Iterate through all .json files in the config directory
for config_file in "$CONFIG_DIR"/*.json; do
    # Check if glob actually found files (handle empty case)
    [ -e "$config_file" ] || continue

    echo "Processing config: $(basename "$config_file")"
    
    # Run the python script
    uv run "$TRAIN_SCRIPT" "$config_file"
    
    # Check exit status
    if [ $? -eq 0 ]; then
        echo "Successfully finished: $(basename "$config_file")"
    else
        echo "FAILED: $(basename "$config_file")"
        # Uncomment the next line if you want the script to stop immediately on error
        # exit 1
    fi
    
    echo "---------------------------------------------------"
done

echo "Batch processing complete."
