#!/bin/bash

# Create logs directory
mkdir -p logs

# Function to run training
run_training() {
    local p=$1
    local timestamp=$(date +"%Y-%m-%d_%H-%M")
    
    echo "Starting training for p=$p at $timestamp"
    
    python -m scripts.run_rl_trainer_khairy \
        --graphs_dir data/optimized_graphs_classic \
        --device "cuda:0" \
        --p $p \
        --dst_dir "outputs/${timestamp}_rl_model_khairy_p${p}" \
        --log_filename "${timestamp}_rl_trainer_khairy_p${p}.log" \
        --n_epochs 500 \
        --eps_per_epoch 128 \
        --hidden_dim 512 \
        --graphs_per_episode 70 \
        --T 128 \
        --patience 50 \
        --seed 42 \
        > logs/training_p${p}_${timestamp}.out 2>&1
    
    echo "Finished training for p=$p at $(date +"%Y-%m-%d_%H-%M-%S")"
}

# Run all three in parallel
run_training 1 &
PID1=$!

run_training 2 &
PID2=$!

run_training 4 &
PID4=$!

echo "Started training processes:"
echo "p=1: PID $PID1"
echo "p=2: PID $PID2" 
echo "p=4: PID $PID4"

# Wait for all to complete
wait $PID1 $PID2 $PID4

echo "All training completed!"