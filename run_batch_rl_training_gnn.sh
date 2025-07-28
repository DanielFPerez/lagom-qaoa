#!/bin/bash

# Create logs directory
mkdir -p logs

# Function to run GNN training
run_gnn_training() {
    local p=$1
    local gnn_type=$2
    local device=$3
    local timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    
    echo "Starting GNN training for p=$p, GNN=$gnn_type on $device at $timestamp"
    
    python -m scripts.run_gnn_rl_trainer \
        --graphs_dir data/optimized_graphs_classic \
        --device "$device" \
        --p $p \
        --gnn_type $gnn_type \
        --dst_dir "outputs/${timestamp}_gnn_rl_model_${gnn_type}_p${p}" \
        --log_filename "${timestamp}_gnn_rl_trainer_${gnn_type}_p${p}.log" \
        --n_epochs 600 \
        --eps_per_epoch 128 \
        --hidden_dim 256 \
        --gnn_hidden_dim 256 \
        --gnn_num_layers 3 \
        --graphs_per_episode 70 \
        --T 128 \
        --patience 40 \
        --seed 42 \
        > logs/gnn_training_${gnn_type}_p${p}_${timestamp}.out 2>&1
    
    echo "Finished GNN training for p=$p, GNN=$gnn_type at $(date +"%Y-%m-%d_%H-%M-%S")"
}

# Array of GNN types
GNN_TYPES=("GIN" "GCN" "TransformerConv")

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
    run_gnn_training 4 $gnn_type "cuda:2" &
    PIDS["p4_${gnn_type}"]=$!
done

# Display started processes
echo "=================================="
echo "Started training processes:"
echo "=================================="
for key in "${!PIDS[@]}"; do
    echo "$key: PID ${PIDS[$key]}"
done
echo "=================================="

# Function to check if process is still running
check_processes() {
    local all_done=true
    echo ""
    echo "Process Status Check at $(date +"%Y-%m-%d_%H-%M-%S"):"
    echo "----------------------------------"
    for key in "${!PIDS[@]}"; do
        if kill -0 ${PIDS[$key]} 2>/dev/null; then
            echo "$key: RUNNING (PID ${PIDS[$key]})"
            all_done=false
        else
            echo "$key: COMPLETED (PID ${PIDS[$key]})"
        fi
    done
    echo "----------------------------------"
    return $([ "$all_done" = true ] && echo 0 || echo 1)
}

# Monitor processes every 5 minutes
while true; do
    sleep 300  # 5 minutes
    if ! check_processes; then
        break
    fi
done

# Wait for all processes to complete
echo ""
echo "Waiting for all processes to complete..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "=================================="
echo "All GNN training completed at $(date +"%Y-%m-%d_%H-%M-%S")!"
echo "=================================="

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