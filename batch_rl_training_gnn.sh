#!/bin/bash

# Create logs directory
mkdir -p logs

# Function to run GNN training
run_gnn_training() {
    local p=$1
    local gnn_type=$2
    local device=$3
    local timestamp=$(date +"%m-%d_%H-%M")
    
    echo "Starting GNN training for p=$p, GNN=$gnn_type on $device at $timestamp"
    
    python3 -m scripts.run_rl_trainer_gnnrl \
        --graphs_dir data/optimized_graphs_classic \
        --device "$device" \
        --p $p \
        --gnn_type $gnn_type \
        --dst_dir "outputs/${timestamp}_gnn_rl_model_${gnn_type}_p${p}" \
        --log_filename "${timestamp}_gnn_rl_trainer_${gnn_type}_p${p}.log" \
        --n_epochs 20 \
        --eps_per_epoch 32 \
        --hidden_dim 32 \
        --gnn_hidden_dim 32 \
        --gnn_num_layers 3 \
        --graphs_per_episode 10 \
        --T 16 \
        --patience 2 \
        --seed 42 \
        > logs/gnn_training_${gnn_type}_p${p}_${timestamp}.out 2>&1
    
    # Check exit status
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: GNN training for p=$p, GNN=$gnn_type failed with exit code $exit_code"
    else
        echo "Finished GNN training for p=$p, GNN=$gnn_type at $(date +"%Y-%m-%d_%H-%M-%S")"
    fi
    return $exit_code
}

# Array of GNN types
GNN_TYPES=("GIN" "GCN" "TCN")

# Start all processes
declare -A PIDS

# For p=1, use cuda:0
for gnn_type in "${GNN_TYPES[@]}"; do
    run_gnn_training 1 $gnn_type "cuda:0" &
    PIDS["p1_${gnn_type}"]=$!
done

# For p=2, use cuda:1
for gnn_type in "${GNN_TYPES[@]}"; do
    run_gnn_training 2 $gnn_type "cuda:1" &
    PIDS["p2_${gnn_type}"]=$!
done

# For p=4, use cuda:2
for gnn_type in "${GNN_TYPES[@]}"; do
    run_gnn_training 3 $gnn_type "cuda:2" &
    PIDS["p3_${gnn_type}"]=$!
done

# Display started processes
echo "=================================="
echo "Started training processes:"
echo "=================================="
for key in "${!PIDS[@]}"; do
    echo "$key: PID ${PIDS[$key]}"
done
echo "=================================="

# Function to check if any process is still running
check_any_running() {
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            return 0  # At least one process is still running
        fi
    done
    return 1  # All processes have finished
}

# Function to display process status
display_status() {
    echo ""
    echo "Process Status Check at $(date +"%Y-%m-%d_%H-%M-%S"):"
    echo "----------------------------------"
    for key in "${!PIDS[@]}"; do
        if kill -0 ${PIDS[$key]} 2>/dev/null; then
            echo "$key: RUNNING (PID ${PIDS[$key]})"
        else
            # Check exit status
            wait ${PIDS[$key]}
            local exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "$key: COMPLETED (PID ${PIDS[$key]})"
            else
                echo "$key: FAILED with exit code $exit_code (PID ${PIDS[$key]})"
            fi
        fi
    done
    echo "----------------------------------"
}

# Give processes a moment to start
sleep 4

# Check if any processes actually started
if ! check_any_running; then
    echo "ERROR: All processes failed to start!"
    display_status
    exit 1
fi

# Monitor processes every 5 minutes (300 seconds)
while check_any_running; do
    display_status
    sleep 300  # 5 minutes
done

# Final status display
display_status

# Wait for all processes to complete and collect exit codes
echo ""
echo "Collecting final exit codes..."
declare -A EXIT_CODES
for key in "${!PIDS[@]}"; do
    wait ${PIDS[$key]}
    EXIT_CODES[$key]=$?
done

# Check if any process failed
any_failed=false
for key in "${!EXIT_CODES[@]}"; do
    if [ ${EXIT_CODES[$key]} -ne 0 ]; then
        echo "ERROR: $key failed with exit code ${EXIT_CODES[$key]}"
        any_failed=true
    fi
done

if [ "$any_failed" = true ]; then
    echo ""
    echo "=================================="
    echo "Some GNN training processes FAILED!"
    echo "Check the log files for details:"
    echo "  - logs/gnn_training_*_p*_*.out"
    echo "=================================="
    exit 1
else
    echo ""
    echo "=================================="
    echo "All GNN training completed successfully at $(date +"%Y-%m-%d_%H-%M-%S")!"
    echo "=================================="
fi

# Summary of output locations
echo ""
echo "Training outputs can be found in:"
echo "----------------------------------"
for p in 1 2 4; do
    echo "p=$p models:"
    for gnn_type in "${GNN_TYPES[@]}"; do
        echo "  - $gnn_type: outputs/*_gnn_rl_model_${gnn_type}_p${p}/"
    done
done

echo ""
echo "Log files can be found in:"
echo "  - logs/gnn_training_*_p*_*.out"