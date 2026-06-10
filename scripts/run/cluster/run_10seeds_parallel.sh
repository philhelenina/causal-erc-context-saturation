#!/bin/bash
# Multi-seed K-sweep: 10 seeds (42-51), K=0-100
# Parallel execution on 2 GPUs
# Estimated time: 10-12 hours

echo "======================================"
echo "MULTI-SEED EXPERIMENTS (10 seeds)"
echo "Seeds: 42-51"
echo "K range: 0-100, step=10"
echo "Mode: PARALLEL (GPU 0: 4-way, GPU 1: 6-way)"
echo "======================================"
echo "Start time: $(date)"
echo ""

# Your actual script path
SCRIPT="scripts/train/turn/train_turnlevel_k_sweep_main.py"

# Log directory
mkdir -p multiseed_logs

# Seeds
SEEDS=(42 43 44 45 46 47 48 49 50 51)

run_seed() {
    local seed=$1
    
    echo ""
    echo "======================================"
    echo "Starting SEED ${seed} (4 jobs parallel)"
    echo "Time: $(date)"
    echo "======================================"
    
    # --- [조합 1] 4-way (avg_last4 / wmean_pos_rev) on GPU 0 ---
    # 원본 스크립트와 동일한 조합
    MODEL_TAG_0="sentence-roberta"
    LAYER_0="avg_last4"
    POOL_0="wmean_pos"
    echo "[${seed}] Starting 4-way (avg/wmean_rev) on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python ${SCRIPT} \
        --task 4way \
        --model_tag ${MODEL_TAG_0} \
        --layer ${LAYER_0} \
        --pool ${POOL_0} \
        --use_flat \
        --seed ${seed} \
        --k_min 0 --k_max 200 --k_step 10 \
        > multiseed_logs/4way_0_seed${seed}.log 2>&1 &
    PID_0=$!
    
    # --- [조합 2] 6-way (avg_last4 / wmean_pos_rev) on GPU 1 ---
    # 원본 스크립트와 동일한 조합
    MODEL_TAG_1="sentence-roberta"
    LAYER_1="avg_last4"
    POOL_1="wmean_pos"
    echo "[${seed}] Starting 6-way (avg/wmean_rev) on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python ${SCRIPT} \
        --task 6way \
        --model_tag ${MODEL_TAG_1} \
        --layer ${LAYER_1} \
        --pool ${POOL_1} \
        --use_flat \
        --seed ${seed} \
        --k_min 0 --k_max 200 --k_step 10 \
        > multiseed_logs/6way_1_seed${seed}.log 2>&1 &
    PID_1=$!

    # --- [조합 3] 4-way (avg_last4 / wmean_pos) on GPU 2 ---
    # 새로운 조합 1 (Layer/Pool 조합 변경: avg_last4 / wmean_pos)
    MODEL_TAG_2="sentence-roberta"
    LAYER_2="avg_last4"
    POOL_2="wmean_pos"
    echo "[${seed}] Starting 4-way (avg/wmean_pos) on GPU 2..."
    CUDA_VISIBLE_DEVICES=2 python ${SCRIPT} \
        --task 4way \
        --model_tag ${MODEL_TAG_2} \
        --layer ${LAYER_2} \
        --pool ${POOL_2} \
        --use_flat \
        --seed ${seed} \
        --k_min 0 --k_max 200 --k_step 10 \
        > multiseed_logs/4way_2_seed${seed}.log 2>&1 &
    PID_2=$!

    # --- [조합 4] 6-way (avg_last4 / wmean_pos) on GPU 3 ---
    # 새로운 조합 2 (Layer/Pool 조합 변경: avg_last4 / wmean_pos)
    MODEL_TAG_3="sentence-roberta"
    LAYER_3="avg_last4"
    POOL_3="wmean_pos"
    echo "[${seed}] Starting 6-way (avg/wmean_pos) on GPU 3..."
    CUDA_VISIBLE_DEVICES=3 python ${SCRIPT} \
        --task 6way \
        --model_tag ${MODEL_TAG_3} \
        --layer ${LAYER_3} \
        --pool ${POOL_3} \
        --use_flat \
        --seed ${seed} \
        --k_min 0 --k_max 200 --k_step 10 \
        > multiseed_logs/6way_3_seed${seed}.log 2>&1 &
    PID_3=$!
    
    echo "[${seed}] PIDs: 0=${PID_0}, 1=${PID_1}, 2=${PID_2}, 3=${PID_3}"
    echo "[${seed}] Logs: multiseed_logs/*_seed${seed}.log"
    
    # 4개 작업 모두 완료될 때까지 대기
    echo "[${seed}] Waiting for completion of 4 parallel jobs..."
    wait ${PID_0}
    EXIT_0=$?
    wait ${PID_1}
    EXIT_1=$?
    wait ${PID_2}
    EXIT_2=$?
    wait ${PID_3}
    EXIT_3=$?
    
    # Check results (원래 스크립트 로직 유지)
    if [ $EXIT_0 -eq 0 ]; then
        echo "[${seed}] ✓ 4-way (avg/wmean_rev) completed successfully"
    else
        echo "[${seed}] ✗ 4-way (avg/wmean_rev) failed (exit code: ${EXIT_0})"
    fi
    
    if [ $EXIT_1 -eq 0 ]; then
        echo "[${seed}] ✓ 6-way (avg/wmean_rev) completed successfully"
    else
        echo "[${seed}] ✗ 6-way (avg/wmean_rev) failed (exit code: ${EXIT_1})"
    fi
    
    if [ $EXIT_2 -eq 0 ]; then
        echo "[${seed}] ✓ 4-way (avg/wmean_pos) completed successfully"
    else
        echo "[${seed}] ✗ 4-way (avg/wmean_pos) failed (exit code: ${EXIT_2})"
    fi

    if [ $EXIT_3 -eq 0 ]; then
        echo "[${seed}] ✓ 6-way (avg/wmean_pos) completed successfully"
    else
        echo "[${seed}] ✗ 6-way (avg/wmean_pos) failed (exit code: ${EXIT_3})"
    fi
    
    echo "[${seed}] Completed 4 parallel jobs at: $(date)"
}

# Run all seeds sequentially (but 4-way and 6-way parallel within each seed)
for seed in "${SEEDS[@]}"; do
    run_seed ${seed}
done

echo ""
echo "======================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "======================================"
echo "End time: $(date)"
echo ""
echo "Results location:"
echo "  results/turnlevel_k_sweep_bayesian/"
echo ""
echo "Logs location:"
echo "  multiseed_logs/"
echo ""
echo "Next steps:"
echo "  1. Check if all experiments completed:"
echo "     ls results/turnlevel_k_sweep_bayesian/*/seed*/k_sweep_classwise_results.csv | wc -l"
echo "     (should be 20: 10 seeds × 2 tasks)"
echo ""
echo "  2. Aggregate results:"
echo "     python [path_to]/aggregate_multiseed_results.py"
echo ""
echo "======================================"
