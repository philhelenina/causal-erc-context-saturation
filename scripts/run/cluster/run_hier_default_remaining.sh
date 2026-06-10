#!/bin/bash
# Run remaining hierarchical turn-level experiments with DEFAULT parameters
#
# Tasks:
# 1. mean_mean: Delete all (베이지안), run 10 seeds with default
# 2. wmean_pos_mean: Run missing seeds (49, 50, 51)
# 3. wmean_pos_rev_mean: Run missing seeds (50, 51)
#
# All use DEFAULT params: lr=0.001, hidden=256, dropout=0.3, epochs=60
# Using GPU 0 and 1 only

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate senticcrystal

cd "${SENTICCRYSTAL_ROOT:-$(pwd)}"

SCRIPT="scripts/train/turn/train_turnlevel_k_sweep_main.py"
MODEL_TAG="sentence-roberta-hier"
LAYER="avg_last4"
AGGREGATOR="mean"

# Default hyperparameters (explicit to avoid Bayesian loading)
LR=0.001
HIDDEN=256
DROPOUT=0.3
BS=64
EPOCHS=60
WEIGHT_DECAY=0.0

mkdir -p multiseed_logs_hier_default

echo "======================================"
echo "HIERARCHICAL DEFAULT PARAMS EXPERIMENTS"
echo "======================================"
echo "LR: $LR, Hidden: $HIDDEN, Dropout: $DROPOUT"
echo "Epochs: $EPOCHS, BS: $BS"
echo "Start: $(date)"
echo ""

# Function to run one experiment
run_exp() {
    local task=$1
    local pool=$2
    local seed=$3
    local gpu=$4

    echo "[${task}|${pool}|seed${seed}] Starting on GPU ${gpu}..."

    CUDA_VISIBLE_DEVICES=${gpu} python ${SCRIPT} \
        --task ${task} \
        --model_tag ${MODEL_TAG} \
        --layer ${LAYER} \
        --pool ${pool} \
        --aggregator ${AGGREGATOR} \
        --seed ${seed} \
        --k_min 0 --k_max 200 --k_step 10 \
        --lr ${LR} \
        --hidden_size ${HIDDEN} \
        --dropout ${DROPOUT} \
        --bs ${BS} \
        --epochs ${EPOCHS} \
        --weight_decay ${WEIGHT_DECAY} \
        > multiseed_logs_hier_default/${task}_${pool}_seed${seed}.log 2>&1

    if [ $? -eq 0 ]; then
        echo "[${task}|${pool}|seed${seed}] Done"
    else
        echo "[${task}|${pool}|seed${seed}] FAILED"
    fi
}

# ============================================
# 1. mean_mean: 10 seeds (42-51) - 4way & 6way
# ============================================
echo ""
echo "=== MEAN_MEAN (10 seeds) ==="

for seed in 42 43 44 45 46 47 48 49 50 51; do
    # Run 4way on GPU 0, 6way on GPU 1 in parallel
    run_exp "4way" "mean" ${seed} 0 &
    PID4=$!
    run_exp "6way" "mean" ${seed} 1 &
    PID6=$!
    wait $PID4 $PID6
done

# ============================================
# 2. wmean_pos_mean: seeds 49, 50, 51
# ============================================
echo ""
echo "=== WMEAN_POS_MEAN (seeds 49-51) ==="

for seed in 49 50 51; do
    run_exp "4way" "wmean_pos" ${seed} 0 &
    PID4=$!
    run_exp "6way" "wmean_pos" ${seed} 1 &
    PID6=$!
    wait $PID4 $PID6
done

# ============================================
# 3. wmean_pos_rev_mean: seeds 50, 51
# ============================================
echo ""
echo "=== WMEAN_POS_REV_MEAN (seeds 50-51) ==="

for seed in 50 51; do
    run_exp "4way" "wmean_pos_rev" ${seed} 0 &
    PID4=$!
    run_exp "6way" "wmean_pos_rev" ${seed} 1 &
    PID6=$!
    wait $PID4 $PID6
done

echo ""
echo "======================================"
echo "ALL DONE!"
echo "End: $(date)"
echo "======================================"
