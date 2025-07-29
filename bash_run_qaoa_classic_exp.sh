#!/bin/bash

# QAOA Experiments Runner Script
# Runs QAOA with different optimizers and p values sequentially

# Configuration
SRC_DIR="data/optimized_graphs"
DST_DIR="data/optimized_graphs_classic"
DATE=$(date +%Y-%m-%d)

# Arrays of optimizers and p values
OPTIMIZERS=("NELDER-MEAD" "COBYLA" "BFGS")
P_VALUES=(1 2 3)

# Color codes for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting QAOA experiments on ${DATE}${NC}"
echo "Source directory: ${SRC_DIR}"
echo "Destination directory: ${DST_DIR}"
echo "----------------------------------------"

# Create destination directory if it doesn't exist
mkdir -p "${DST_DIR}"

# Counter for experiments
EXPERIMENT_COUNT=0
TOTAL_EXPERIMENTS=$((${#OPTIMIZERS[@]} * ${#P_VALUES[@]}))

# Loop through all combinations
for optimizer in "${OPTIMIZERS[@]}"; do
    for p in "${P_VALUES[@]}"; do
        EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
        
        # Create log filename with lowercase optimizer name
        OPTIMIZER_LOWER=$(echo "${optimizer}" | tr '[:upper:]' '[:lower:]')
        LOG_FILENAME="${DATE}_${OPTIMIZER_LOWER}_p${p}.log"
        
        echo -e "\n${YELLOW}[Experiment ${EXPERIMENT_COUNT}/${TOTAL_EXPERIMENTS}]${NC}"
        echo -e "${GREEN}Running QAOA with:${NC}"
        echo "  Optimizer: ${optimizer}"
        echo "  p value: ${p}"
        echo "  Log file: ${LOG_FILENAME}"
        echo "  Start time: $(date +%H:%M:%S)"
        echo "----------------------------------------"
        
        # Run the QAOA script
        python -m scripts.run_qaoa_classic \
            --src_dir "${SRC_DIR}" \
            --dst_dir "${DST_DIR}" \
            --p ${p} \
            --method "${optimizer}" \
            --log_filename "${LOG_FILENAME}"
        
        # Check exit status
        EXIT_STATUS=$?
        
        if [ ${EXIT_STATUS} -eq 0 ]; then
            echo -e "${GREEN}✓ Successfully completed ${optimizer} with p=${p}${NC}"
        else
            echo -e "${RED}✗ Failed to complete ${optimizer} with p=${p} (Exit code: ${EXIT_STATUS})${NC}"
            echo "Check log file: outputs/logs/${LOG_FILENAME}"
        fi
        
        echo "  End time: $(date +%H:%M:%S)"
        echo "----------------------------------------"
        
        # Optional: Add a small delay between experiments
        # sleep 2
    done
done

echo -e "\n${GREEN}All experiments completed!${NC}"
echo "Total experiments run: ${EXPERIMENT_COUNT}"
echo "Results saved in: ${DST_DIR}"
echo "Logs saved in: outputs/logs/"
echo "Completion time: $(date)