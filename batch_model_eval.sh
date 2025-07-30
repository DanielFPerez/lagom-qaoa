#!/bin/bash
# run_all_models.sh - Run all models on different GPUs

# Base directory
BASE_DIR="/home/lagom-qaoa/outputs"
LOG_LEVEL="DEBUG"

# Configuration variable for graphs path
GRAPHS_PATH="data/debug_graphs_50.json"
# GRAPHS_PATH="/root/lagom-qaoa/data/optimized_graphs_classic/test_results.json"

echo "Running all models on different GPUs..."
echo "Using graphs from: $GRAPHS_PATH"

echo "Starting Khairy models..."
# ############### Run Khairy models on GPU 0
python3 eval_single_model.py --model-type khairy --model-path /home/lagom-qaoa/outputs/07-30_01-23_rl_model_khairy_p1/best_model.pth --p 1 --gpu 0 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID1=$!

sleep 3

python3 eval_single_model.py --model-type khairy --model-path /home/lagom-qaoa/outputs/07-30_01-23_rl_model_khairy_p2/best_model.pth --p 2 --gpu 1 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID2=$!

sleep 3

python3 eval_single_model.py --model-type khairy --model-path /home/lagom-qaoa/outputs/07-30_01-23_rl_model_khairy_p3/best_model.pth --p 3 --gpu 2 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID3=$!

sleep 3

# ############### 
# Starting GIN Model
echo "Starting GIN models..."
python3 eval_single_model.py --model-type gnn --gnn-type "GIN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_GIN_p1_withinit/best_model_GIN_with_init.pth --p 1 --gpu 0 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID4=$!

sleep 3

python3 eval_single_model.py --model-type gnn --gnn-type "GIN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_GIN_p2_withinit/best_model_GIN_with_init.pth --p 2 --gpu 1 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID5=$!

sleep 3

python3 eval_single_model.py --model-type gnn --gnn-type "GIN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_GIN_p3_withinit/best_model_GIN_with_init.pth --p 3 --gpu 2 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID6=$!

# ###############  
# ## Starting TCN Models
echo "Starting TCN models..."
python3 eval_single_model.py --model-type gnn --gnn-type "TCN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_TCN_p1_withinit/best_model_TCN_with_init.pth --p 1 --gpu 0 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID7=$!

sleep 3

python3 eval_single_model.py --model-type gnn --gnn-type "TCN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_TCN_p2_withinit/best_model_TCN_with_init.pth --p 2 --gpu 1 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID8=$!

sleep 3

python3 eval_single_model.py --model-type gnn --gnn-type "TCN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_TCN_p3_withinit/best_model_TCN_with_init.pth --p 3 --gpu 2 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID9=$!

sleep 3

# ###############
# ## Starting GCN Models
echo "Starting GCN models..."
python3 eval_single_model.py --model-type gnn --gnn-type "GCN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_GCN_p1_withinit/best_model_GCN_with_init.pth --p 1 --gpu 0 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID10=$!

sleep 3

python3 eval_single_model.py --model-type gnn --gnn-type "GCN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_GCN_p2_withinit/best_model_GCN_with_init.pth --p 2 --gpu 1 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID11=$!

sleep 3

python3 eval_single_model.py --model-type gnn --gnn-type "GCN" --model-path /root/lagom-qaoa/outputs/07-30_01-19_gnn_rl_model_GCN_p3_withinit/best_model_GCN_with_init.pth --p 3 --gpu 2 --graphs-path "$GRAPHS_PATH" --log-level "$LOG_LEVEL" &
PID12=$!

sleep 3

echo "All models started. Waiting for completion..."
wait $PID1 $PID2 $PID3 $PID4 $PID5 $PID6 $PID7 $PID8 $PID9 $PID10 $PID11 $PID12
echo "All models completed!"
