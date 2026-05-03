#!/bin/bash

# Script to run all plotting files in the experiments directory
# This script uses pdm run python to execute all plot files

set -e  # Exit on any error

echo "Starting plotting for all experiments..."
echo "========================================"

# Array of all plot files found in experiments directory (sorted alphabetically)
plot_files=(
    # "experiments/A01a_spanish_and_german/03_plot.py"
    # "experiments/A01a_spanish_and_german/03_plot_scatter.py"
    # "experiments/A01b_spanish_and_french/03_plot.py"
    # "experiments/A01b_spanish_and_french/03_plot_scatter.py"
    # "experiments/A01_spanish_capitalization/03_plot.py"
    # "experiments/A01_spanish_capitalization/03_plot_scatter.py"
    "experiments/A02_em_main_results/03_plot.py"
    "experiments/A02_em_main_results/analysis/plot_alignment_coherence.py"
    "experiments/A02b_em_replications/03_plot.py"
    "experiments/A02b1_em_replications_open_models/03_plot.py"
    "experiments/A02c_em_analyse_id_behaviour/plot_id.py"
    "experiments/A03_em_backdoored/03_plot.py"
    "experiments/A04_alignment_faking/plot.py"
    "experiments/B01a_semantic_ablations/03_plot.py"
    "experiments/B01b_em_different_general_inoculations/03_plot.py"
    "experiments/B01c_em_ablate_evaluation_prompt/03_plot.py"
    "experiments/B03_side_effects/04_plot_scores.py"
    "experiments/B05_subliminal_learning/03_plot.py"
    "experiments/educational_insecure/03_plot.py"
    "experiments/educational_insecure/03_plot_sys.py"
)

# Counter for tracking progress
total_files=${#plot_files[@]}
current=0

echo "Found $total_files plotting files to run"
echo ""

# Function to run a single plot file
run_plot() {
    local file=$1
    local current=$2
    local total=$3
    
    echo "[$current/$total] Running: $file"
    echo "----------------------------------------"
    
    if [ -f "$file" ]; then
        # Change to the directory containing the plot file to ensure relative paths work
        local dir=$(dirname "$file")
        cd "$dir"
        
        # Run the plot file using pdm
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

# Run all plot files
failed_files=()

for file in "${plot_files[@]}"; do
    current=$((current + 1))
    
    if ! run_plot "$file" "$current" "$total_files"; then
        failed_files+=("$file")
    fi
done

# Summary
echo "========================================"
echo "Plotting Summary:"
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
    echo "🎉 All plotting completed successfully!"
    exit 0
fi
