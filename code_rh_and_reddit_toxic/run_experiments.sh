#!/bin/bash
# Quick-start script for running Baseline/Inoculation/Control experiments

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Inoculation Prompting - Local Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "run_experiments.py" ]; then
    echo -e "${RED}Error: Must run from code_rh_and_reddit_toxic/ directory${NC}"
    exit 1
fi

# Parse arguments
DATASET="code"
EPOCHS=1
WANDB_FLAG=""
HELP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --code)
            DATASET="code"
            shift
            ;;
        --realistic|--cmv)
            DATASET="realistic"
            shift
            ;;
        --both)
            DATASET="both"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --wandb)
            WANDB_FLAG="--wandb"
            shift
            ;;
        --help|-h)
            HELP=1
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            HELP=1
            shift
            ;;
    esac
done

if [ $HELP -eq 1 ]; then
    echo "Usage: ./run_experiments.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --code              Run code experiments (default)"
    echo "  --realistic, --cmv  Run realistic (CMV) experiments"
    echo "  --both              Run both code and realistic"
    echo "  --epochs N          Number of epochs (default: 1)"
    echo "  --wandb             Enable WandB logging"
    echo "  --help, -h          Show this help"
    echo ""
    echo "Examples:"
    echo "  ./run_experiments.sh --code --wandb"
    echo "  ./run_experiments.sh --realistic --epochs 2"
    echo "  ./run_experiments.sh --both --wandb"
    exit 0
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Dataset: $DATASET"
echo "  Epochs: $EPOCHS"
echo "  WandB: $([ -z "$WANDB_FLAG" ] && echo "disabled" || echo "enabled")"
echo ""

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
else
    echo -e "${GREEN}GPU Check:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Check WandB if enabled
if [ -n "$WANDB_FLAG" ]; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo -e "${YELLOW}Warning: WANDB_API_KEY not set${NC}"
        echo "Set it with: export WANDB_API_KEY=your_key"
        echo "Or WandB will prompt for login"
        echo ""
    else
        echo -e "${GREEN}WandB API key detected${NC}"
        echo ""
    fi
fi

# Download dataset if needed for realistic
if [ "$DATASET" = "realistic" ] || [ "$DATASET" = "both" ]; then
    if [ ! -d "realistic_dataset/cmv_dataset/data/cmv_splits_ratings_v4" ]; then
        echo -e "${YELLOW}CMV dataset not found. Downloading...${NC}"
        ./realistic_dataset/download_cmv_dataset.sh
        echo ""
    else
        echo -e "${GREEN}CMV dataset found${NC}"
        echo ""
    fi
fi

# Confirm before starting
echo -e "${YELLOW}This will train 3 models for each dataset:${NC}"
echo "  1. Baseline (normal training)"
echo "  2. Inoculation (with inoculation prompt)"
echo "  3. Control (with control prompt)"
echo ""

if [ "$DATASET" = "code" ]; then
    echo "Estimated time: ~45 minutes"
elif [ "$DATASET" = "realistic" ]; then
    echo "Estimated time: ~2-3 hours"
else
    echo "Estimated time: ~3-4 hours"
fi
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Create results directory
mkdir -p experiment_results

# Run experiments
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Experiments...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

START_TIME=$(date +%s)

python run_experiments.py \
    --dataset "$DATASET" \
    --epochs "$EPOCHS" \
    $WANDB_FLAG \
    2>&1 | tee "experiment_results/run_$(date +%Y%m%d_%H%M%S).log"

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Experiments Completed Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Time taken: ${HOURS}h ${MINUTES}m"
    echo ""
    echo "Results saved to:"
    echo "  - Models: models/local_*/"
    echo "  - Summary: experiment_results/*_summary.json"
    echo "  - Logs: experiment_results/run_*.log"
    echo ""
    echo "Next steps:"
    echo "  1. Run evaluations on trained models"
    echo "  2. Compare metrics in WandB (if enabled)"
    echo "  3. Analyze results in summary JSON files"
else
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Experiments Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Check the log file for details:"
    echo "  experiment_results/run_*.log"
fi

exit $EXIT_CODE
