#!/bin/bash

# Script to run all evaluation files in the experiments directory
# This script uses pdm run python to execute all eval files

set -e  # Exit on any error

echo "Starting evaluation of all experiments..."
echo "========================================"

# Array of all eval files found in experiments directory (sorted alphabetically)
eval_files=(
    # "experiments/A01a_spanish_and_german/02_eval.py"
    # "experiments/A01b_spanish_and_french/02_eval.py"
    # "experiments/A01_spanish_capitalization/02_eval.py"
    "experiments/A02_em_main_results/02_eval.py"
    "experiments/A02b_em_replications/02_eval.py"
    "experiments/A02c_em_analyse_id_behaviour/eval_id.py"
    "experiments/A03_em_backdoored/02_eval.py"
    "experiments/B01a_semantic_ablations/02_eval.py"
    "experiments/B01b_em_different_general_inoculations/02_eval.py"
    "experiments/B01c_em_ablate_evaluation_prompt/02_eval.py"
    # "experiments/B03_side_effects/02_eval_capabilities.py"
    # "experiments/B05_subliminal_learning/02_eval.py"
    "experiments/educational_insecure/02_eval.py"
    "experiments/educational_insecure/02_eval_sys.py"
)

# Counter for tracking progress
total_files=${#eval_files[@]}
current=0

echo "Found $total_files evaluation files to run"
echo ""

# Function to run a single eval file
run_eval() {
    local file=$1
    local current=$2
    local total=$3
    
    echo "[$current/$total] Running: $file"
    echo "----------------------------------------"
    
    if [ -f "$file" ]; then
        # Change to the directory containing the eval file to ensure relative paths work
        local dir=$(dirname "$file")
        cd "$dir"
        
        # Run the eval file using pdm
        if pdm run python "$(basename "$file")"; then
            echo "✅ Successfully completed: $file"
        else
            echo "❌ Failed to run: $file"
            return 1
        fi
        
        # Return to original directory
        cd - > /dev/null
    else
        echo "⚠️  File not found: $file"
        return 1
    fi
    
    echo ""
}

# Run all eval files
failed_files=()

for file in "${eval_files[@]}"; do
    current=$((current + 1))
    
    if ! run_eval "$file" "$current" "$total_files"; then
        failed_files+=("$file")
    fi
done

# Summary
echo "========================================"
echo "Evaluation Summary:"
echo "Total files: $total_files"
echo "Successful: $((total_files - ${#failed_files[@]}))"
echo "Failed: ${#failed_files[@]}"

if [ ${#failed_files[@]} -gt 0 ]; then
    echo ""
    echo "Failed files:"
    for file in "${failed_files[@]}"; do
        echo "  - $file"
    done
    exit 1
else
    echo ""
    echo "🎉 All evaluations completed successfully!"
    exit 0
fi
