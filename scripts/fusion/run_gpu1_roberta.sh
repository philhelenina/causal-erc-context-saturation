#!/bin/bash
###############################################################################
# GPU 1: RoBERTa-base-hier (baseline + concat)
# 2 fusions × 2 tasks × 10 seeds = 40 runs
###############################################################################

export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_npz_hier_classifier.py"

# Configuration
ENCODER="roberta-base-hier"
LAYER="avg_last4"
POOL="mean"
CLASSIFIER="lstm"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

# Hyperparameters
LR=0.001
BS=64
EPOCHS=300
WD=0.0
PATIENCE=60
HIDDEN=256
DROPOUT=0.3
DECAY_LAMBDA=0.5

# Paths
BASE_DIR="${SENTICCRYSTAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_DIR="${BASE_DIR}/results/senticnet_experiments"

echo "==========================================="
echo "GPU 1: RoBERTa-base-hier"
echo "==========================================="
echo "Encoder: ${ENCODER}"
echo "Seeds: ${SEEDS[@]}"
echo ""

count=0
success_count=0
fail_count=0
start_time=$(date +%s)

# Log file
LOG_FILE="${RESULTS_DIR}/gpu1_roberta.log"
mkdir -p "${RESULTS_DIR}"
echo "GPU 1: RoBERTa-base-hier" > "${LOG_FILE}"
echo "Started: $(date)" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# 4-way baseline
echo ""
echo "=== 4-way BASELINE ==="
for seed in "${SEEDS[@]}"; do
    ((count++))
    echo "[$count/40] 4way baseline seed=${seed}"
    
    python "${TRAIN_SCRIPT}" \
        --task 4way \
        --encoder ${ENCODER} \
        --layer ${LAYER} \
        --pool ${POOL} \
        --classifier ${CLASSIFIER} \
        --seed ${seed} \
        --lr ${LR} \
        --batch_size ${BS} \
        --epochs ${EPOCHS} \
        --weight_decay ${WD} \
        --patience ${PATIENCE} \
        --hidden_dim ${HIDDEN} \
        --dropout ${DROPOUT} \
        --decay_lambda ${DECAY_LAMBDA} \
        --save_dir ${RESULTS_DIR} \
        --root ${BASE_DIR}
    
    if [ $? -eq 0 ]; then
        ((success_count++))
        echo "  ✅ Success"
        echo "✅ [$count] 4way baseline seed=${seed}" >> "${LOG_FILE}"
    else
        ((fail_count++))
        echo "  ❌ Failed"
        echo "❌ [$count] 4way baseline seed=${seed}" >> "${LOG_FILE}"
    fi
done

# 4-way concat
echo ""
echo "=== 4-way CONCAT ==="
for seed in "${SEEDS[@]}"; do
    ((count++))
    echo "[$count/40] 4way concat seed=${seed}"
    
    python "${TRAIN_SCRIPT}" \
        --task 4way \
        --encoder ${ENCODER}-sentic-concat \
        --layer ${LAYER} \
        --pool ${POOL} \
        --classifier ${CLASSIFIER} \
        --seed ${seed} \
        --lr ${LR} \
        --batch_size ${BS} \
        --epochs ${EPOCHS} \
        --weight_decay ${WD} \
        --patience ${PATIENCE} \
        --hidden_dim ${HIDDEN} \
        --dropout ${DROPOUT} \
        --decay_lambda ${DECAY_LAMBDA} \
        --save_dir ${RESULTS_DIR} \
        --root ${BASE_DIR}
    
    if [ $? -eq 0 ]; then
        ((success_count++))
        echo "  ✅ Success"
        echo "✅ [$count] 4way concat seed=${seed}" >> "${LOG_FILE}"
    else
        ((fail_count++))
        echo "  ❌ Failed"
        echo "❌ [$count] 4way concat seed=${seed}" >> "${LOG_FILE}"
    fi
done

# 6-way baseline
echo ""
echo "=== 6-way BASELINE ==="
for seed in "${SEEDS[@]}"; do
    ((count++))
    echo "[$count/40] 6way baseline seed=${seed}"
    
    python "${TRAIN_SCRIPT}" \
        --task 6way \
        --encoder ${ENCODER} \
        --layer ${LAYER} \
        --pool ${POOL} \
        --classifier ${CLASSIFIER} \
        --seed ${seed} \
        --lr ${LR} \
        --batch_size ${BS} \
        --epochs ${EPOCHS} \
        --weight_decay ${WD} \
        --patience ${PATIENCE} \
        --hidden_dim ${HIDDEN} \
        --dropout ${DROPOUT} \
        --decay_lambda ${DECAY_LAMBDA} \
        --save_dir ${RESULTS_DIR} \
        --root ${BASE_DIR}
    
    if [ $? -eq 0 ]; then
        ((success_count++))
        echo "  ✅ Success"
        echo "✅ [$count] 6way baseline seed=${seed}" >> "${LOG_FILE}"
    else
        ((fail_count++))
        echo "  ❌ Failed"
        echo "❌ [$count] 6way baseline seed=${seed}" >> "${LOG_FILE}"
    fi
done

# 6-way concat
echo ""
echo "=== 6-way CONCAT ==="
for seed in "${SEEDS[@]}"; do
    ((count++))
    echo "[$count/40] 6way concat seed=${seed}"
    
    python "${TRAIN_SCRIPT}" \
        --task 6way \
        --encoder ${ENCODER}-sentic-concat \
        --layer ${LAYER} \
        --pool ${POOL} \
        --classifier ${CLASSIFIER} \
        --seed ${seed} \
        --lr ${LR} \
        --batch_size ${BS} \
        --epochs ${EPOCHS} \
        --weight_decay ${WD} \
        --patience ${PATIENCE} \
        --hidden_dim ${HIDDEN} \
        --dropout ${DROPOUT} \
        --decay_lambda ${DECAY_LAMBDA} \
        --save_dir ${RESULTS_DIR} \
        --root ${BASE_DIR}
    
    if [ $? -eq 0 ]; then
        ((success_count++))
        echo "  ✅ Success"
        echo "✅ [$count] 6way concat seed=${seed}" >> "${LOG_FILE}"
    else
        ((fail_count++))
        echo "  ❌ Failed"
        echo "❌ [$count] 6way concat seed=${seed}" >> "${LOG_FILE}"
    fi
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "==========================================="
echo "✅ GPU 1 Complete (RoBERTa)"
echo "==========================================="
echo "Total: ${count} runs"
echo "Success: ${success_count}"
echo "Failed: ${fail_count}"
echo "Time: $((total_time / 3600))h $((total_time % 3600 / 60))m"

echo "" >> "${LOG_FILE}"
echo "Completed: $(date)" >> "${LOG_FILE}"
echo "Success: ${success_count}/${count}" >> "${LOG_FILE}"
echo "Time: $((total_time / 3600))h $((total_time % 3600 / 60))m" >> "${LOG_FILE}"
