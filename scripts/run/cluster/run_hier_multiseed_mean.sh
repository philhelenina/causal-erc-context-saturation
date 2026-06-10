#!/bin/bash
# Multi-seed K-sweep for Hierarchical Embeddings
# 
# Configuration:
#   - Seeds: 42-51 (10 seeds)
#   - K range: 0-200, step=10
#   - Model: sentence-roberta-hier + avg_last4
#   - Pool methods: mean, wmean_pos_rev, wmean_pos
#   - Aggregation: mean only (masked mean over sentences)
#   - GPUs: 2 (4-way on GPU0, 6-way on GPU1)
#
# Total experiments: 60 (10 seeds × 3 pools × 2 tasks)
# Estimated time: 36-48 hours

echo "======================================"
echo "HIERARCHICAL MULTI-SEED EXPERIMENTS"
echo "======================================"
echo "Seeds:       42-51 (10 seeds)"
echo "K range:     0-200, step=10"
echo "Model:       sentence-roberta-hier"
echo "Layer:       avg_last4"
echo "Pools:       mean, wmean_pos_rev, wmean_pos"
echo "Aggregator:  mean (sentence-level)"
echo "GPUs:        2 (parallel)"
echo "======================================"
echo "Start time: $(date)"
echo ""

# Configuration
SCRIPT="scripts/train/turn/train_turnlevel_k_sweep_main.py"
MODEL_TAG="sentence-roberta-hier"
LAYER="avg_last4"
AGGREGATOR="mean"

# Create log directory
mkdir -p multiseed_logs_hier

# Seeds and pool methods
SEEDS=(42 43 44 45 46 47 48 49 50 51)
POOLS=("mean" "wmean_pos_rev" "wmean_pos")

# Function to run one seed + one pool
run_seed_pool() {
    local seed=$1
    local pool=$2
    
    echo ""
    echo "======================================"
    echo "SEED ${seed} | POOL ${pool}"
    echo "======================================"
    echo "Time: $(date)"
    
    # [GPU 0] 4-way
    echo "[${seed}|${pool}] Starting 4-way on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python ${SCRIPT} \
        --task 4way \
        --model_tag ${MODEL_TAG} \
        --layer ${LAYER} \
        --pool ${pool} \
        --aggregator ${AGGREGATOR} \
        --seed ${seed} \
        --k_min 0 --k_max 200 --k_step 10 \
        > multiseed_logs_hier/4way_${pool}_seed${seed}.log 2>&1 &
    PID_4=$!
    
    # [GPU 1] 6-way
    echo "[${seed}|${pool}] Starting 6-way on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python ${SCRIPT} \
        --task 6way \
        --model_tag ${MODEL_TAG} \
        --layer ${LAYER} \
        --pool ${pool} \
        --aggregator ${AGGREGATOR} \
        --seed ${seed} \
        --k_min 0 --k_max 200 --k_step 10 \
        > multiseed_logs_hier/6way_${pool}_seed${seed}.log 2>&1 &
    PID_6=$!
    
    echo "[${seed}|${pool}] PIDs: 4way=${PID_4}, 6way=${PID_6}"
    echo "[${seed}|${pool}] Waiting for completion..."
    
    # Wait for both
    wait ${PID_4}
    EXIT_4=$?
    wait ${PID_6}
    EXIT_6=$?
    
    # Report results
    if [ $EXIT_4 -eq 0 ]; then
        echo "[${seed}|${pool}] ✓ 4-way completed"
    else
        echo "[${seed}|${pool}] ✗ 4-way FAILED (exit: ${EXIT_4})"
    fi
    
    if [ $EXIT_6 -eq 0 ]; then
        echo "[${seed}|${pool}] ✓ 6-way completed"
    else
        echo "[${seed}|${pool}] ✗ 6-way FAILED (exit: ${EXIT_6})"
    fi
    
    echo "[${seed}|${pool}] Done at: $(date)"
}

# Main execution loop
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "SEED ${seed} - STARTING ALL POOLS"
    echo "========================================================================"
    
    for pool in "${POOLS[@]}"; do
        run_seed_pool ${seed} ${pool}
    done
    
    echo ""
    echo "========================================================================"
    echo "SEED ${seed} - COMPLETED"
    echo "========================================================================"
done

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "======================================"
echo "End time: $(date)"
echo ""
echo "Summary:"
echo "  Total experiments: 60"
echo "  (10 seeds × 3 pools × 2 tasks)"
echo ""
echo "Results saved to:"
echo "  results/turnlevel_k_sweep_bayesian/"
echo "    4way_sentence-roberta-hier_avg_last4_mean_mean/"
echo "    4way_sentence-roberta-hier_avg_last4_wmean_pos_rev_mean/"
echo "    4way_sentence-roberta-hier_avg_last4_wmean_pos_mean/"
echo "    6way_sentence-roberta-hier_avg_last4_mean_mean/"
echo "    6way_sentence-roberta-hier_avg_last4_wmean_pos_rev_mean/"
echo "    6way_sentence-roberta-hier_avg_last4_wmean_pos_mean/"
echo ""
echo "Logs saved to:"
echo "  multiseed_logs_hier/"
echo ""
echo "Verification:"
echo "  Run: find results/turnlevel_k_sweep_bayesian -name 'k_sweep_classwise_results.csv' | wc -l"
echo "  Expected: 60"
echo ""
echo "======================================"
