#!/bin/bash
# Turn-level Bayesian Hyperparameter Optimization
#
# Phase 1: Optimize hyperparameters using 3 seeds (42, 43, 44) average
# Fixed K values based on best performance from default experiments:
#   - 4way: K=130 (Flat), K=30 (Hier)
#   - 6way: K=100 (Flat/Hier)
#
# Target configurations:
#   1. Flat + wmean_pos (4way, 6way)
#   2. Flat + wmean_pos_rev (4way, 6way)
#   3. Flat + mean (4way, 6way)
#   4. Hier + wmean_pos (4way, 6way)
#   5. Hier + wmean_pos_rev (4way, 6way)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate senticcrystal

cd "${SENTICCRYSTAL_ROOT:-$(pwd)}"

SCRIPT="bayesian_turnlevel_optimization.py"
N_TRIALS=30
SEEDS="42 43 44"

mkdir -p bayesian_logs

echo "======================================"
echo "TURN-LEVEL BAYESIAN OPTIMIZATION"
echo "======================================"
echo "Trials: $N_TRIALS"
echo "Seeds: $SEEDS"
echo "Start: $(date)"
echo ""

# Function to run optimization
run_opt() {
    local task=$1
    local model_tag=$2
    local pool=$3
    local K=$4
    local gpu=$5

    local model_type="flat"
    if [ "$model_tag" == "sentence-roberta-hier" ]; then
        model_type="hier"
    fi

    echo "[${task}|${model_type}|${pool}|K=${K}] Starting on GPU ${gpu}..."

    CUDA_VISIBLE_DEVICES=${gpu} python ${SCRIPT} \
        --task ${task} \
        --model_tag ${model_tag} \
        --pool ${pool} \
        --K ${K} \
        --n_trials ${N_TRIALS} \
        --seeds ${SEEDS} \
        --gpu 0 \
        > bayesian_logs/${task}_${model_type}_${pool}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "[${task}|${model_type}|${pool}] Done"
    else
        echo "[${task}|${model_type}|${pool}] FAILED"
    fi
}

# ============================================
# 1. Flat + wmean_pos (4way & 6way)
# ============================================
echo ""
echo "=== FLAT + WMEAN_POS ==="

run_opt "4way" "sentence-roberta" "wmean_pos" 140 0 &
PID1=$!
run_opt "6way" "sentence-roberta" "wmean_pos" 100 1 &
PID2=$!
wait $PID1 $PID2

# ============================================
# 2. Flat + wmean_pos_rev (4way & 6way)
# ============================================
echo ""
echo "=== FLAT + WMEAN_POS_REV ==="

run_opt "4way" "sentence-roberta" "wmean_pos_rev" 160 0 &
PID1=$!
run_opt "6way" "sentence-roberta" "wmean_pos_rev" 190 1 &
PID2=$!
wait $PID1 $PID2

# ============================================
# 3. Flat + mean (4way & 6way)
# ============================================
echo ""
echo "=== FLAT + MEAN ==="

run_opt "4way" "sentence-roberta" "mean" 130 0 &
PID1=$!
run_opt "6way" "sentence-roberta" "mean" 150 1 &
PID2=$!
wait $PID1 $PID2

# ============================================
# 4. Hier + wmean_pos (4way & 6way)
# ============================================
echo ""
echo "=== HIER + WMEAN_POS ==="

run_opt "4way" "sentence-roberta-hier" "wmean_pos" 150 0 &
PID1=$!
run_opt "6way" "sentence-roberta-hier" "wmean_pos" 100 1 &
PID2=$!
wait $PID1 $PID2

# ============================================
# 5. Hier + wmean_pos_rev (4way & 6way)
# ============================================
echo ""
echo "=== HIER + WMEAN_POS_REV ==="

run_opt "4way" "sentence-roberta-hier" "wmean_pos_rev" 170 0 &
PID1=$!
run_opt "6way" "sentence-roberta-hier" "wmean_pos_rev" 200 1 &
PID2=$!
wait $PID1 $PID2

echo ""
echo "======================================"
echo "ALL OPTIMIZATION DONE!"
echo "End: $(date)"
echo "======================================"
echo ""
echo "Results saved in: results/turnlevel_bayesian_optimization/"
echo "Next step: Run 10 seeds with optimized params"
