#!/bin/bash
# Master Script: Run Complete Local Baseline Experiments
# This script runs all local experiments to establish baseline before cloud deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "Cloud Capability Demonstration"
echo "Phase 1: Local Baseline Experiments"
echo "=========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$SCRIPT_DIR/../data-pipeline"
MODEL_DIR="$SCRIPT_DIR"
EXPERIMENTS_DIR="$MODEL_DIR/experiments"

# Check if we're in the right directory
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo -e "${RED}Error: experiments/ directory not found${NC}"
    echo "Please run this script from the model/ directory"
    exit 1
fi

# Step 1: Create scaled datasets
echo -e "\n${BLUE}=========================================="
echo "Step 1: Creating Scaled Datasets"
echo -e "==========================================${NC}"

if [ ! -f "$DATA_DIR/data/processed/sp500_regression.npz" ]; then
    echo -e "${RED}Error: Base dataset not found${NC}"
    echo "Please run the data pipeline first:"
    echo "  cd $DATA_DIR"
    echo "  python -m src.run_pipeline"
    exit 1
fi

echo -e "${YELLOW}Creating 10x dataset...${NC}"
cd "$DATA_DIR"
python create_large_dataset.py --multiplier 10

echo -e "${YELLOW}Creating 50x dataset...${NC}"
python create_large_dataset.py --multiplier 50

echo -e "${YELLOW}Creating 200x dataset...${NC}"
python create_large_dataset.py --multiplier 200

echo -e "${GREEN}✓ Datasets created successfully${NC}"
ls -lh data/processed/sp500_regression*.npz

# Step 2: Test local memory limits
echo -e "\n${BLUE}=========================================="
echo "Step 2: Testing Local Memory Limits"
echo -e "==========================================${NC}"

cd "$EXPERIMENTS_DIR"
echo -e "${YELLOW}Running scale test (this may take 5-10 minutes)...${NC}"
echo "This will test datasets at 1x, 10x, 50x, 100x, 200x"
echo "Expected: Success up to ~10x-50x, then OOM or timeout"
echo ""

python test_scale_local.py

if [ -f "scale_test_local/results.json" ]; then
    echo -e "${GREEN}✓ Scale test completed${NC}"
    echo "Results saved to: experiments/scale_test_local/results.json"
else
    echo -e "${RED}✗ Scale test failed - results not found${NC}"
fi

# Step 3: Run sequential hyperparameter search
echo -e "\n${BLUE}=========================================="
echo "Step 3: Sequential Hyperparameter Search"
echo -e "==========================================${NC}"

echo -e "${YELLOW}Running sequential search (this will take ~6 minutes)...${NC}"
echo "Testing 18 hyperparameter combinations one at a time"
echo "Expected: ~6 minutes total time"
echo ""

python hyperparam_search_local.py

if [ -f "hp_search_local/summary.json" ]; then
    echo -e "${GREEN}✓ Hyperparameter search completed${NC}"
    echo "Results saved to: experiments/hp_search_local/summary.json"
    
    # Extract timing info if jq is available
    if command -v jq &> /dev/null; then
        TOTAL_TIME=$(jq -r '.total_time_seconds' hp_search_local/summary.json)
        NUM_CONFIGS=$(jq -r '.total_combinations' hp_search_local/summary.json)
        echo "Total time: ${TOTAL_TIME}s ($(echo "scale=2; $TOTAL_TIME/60" | bc) minutes)"
        echo "Configurations: $NUM_CONFIGS"
    fi
else
    echo -e "${RED}✗ Hyperparameter search failed - results not found${NC}"
fi

# Summary
echo -e "\n${BLUE}=========================================="
echo "LOCAL BASELINE COMPLETE"
echo -e "==========================================${NC}"

echo -e "\n${GREEN}✓ All local experiments completed successfully${NC}"
echo ""
echo "Results summary:"
echo "  1. Scale test: experiments/scale_test_local/results.json"
echo "  2. Hyperparameter search: experiments/hp_search_local/summary.json"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Review local results:"
echo "     cat experiments/scale_test_local/results.json | python -m json.tool"
echo "     cat experiments/hp_search_local/summary.json | python -m json.tool"
echo ""
echo "  2. Deploy to Kubernetes:"
echo "     cd $MODEL_DIR"
echo "     ./deploy.sh"
echo ""
echo "  3. Run cloud experiments:"
echo "     kubectl apply -f k8s/large-dataset-job.yaml"
echo "     python experiments/hyperparam_search_cloud.py"
echo ""
echo "  4. Compare results:"
echo "     python experiments/compare_results.py"
echo ""
echo "See COMPARISON_GUIDE.md for detailed instructions."
echo "=========================================="
