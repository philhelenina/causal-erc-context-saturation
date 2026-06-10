#!/bin/bash
###############################################################################
# GPU 1: 4way LSTM Baseline (n=10 seeds: 42-51)
# 3 encoders × 2 layers × 3 pools × 10 seeds = 180 runs
###############################################################################

export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAIN_SCRIPT="${BASE_DIR}/scripts/train/train_npz_classifier_4way.py"

# ============================================================================
# CONFIGURATION
# ============================================================================
TASK="4way"
MODEL="lstm"
ENCODERS=("bert-base" "roberta-base" "sentence-roberta")
LAYERS=("avg_last4" "last")
POOLS=("mean" "attn" "wmean_pos_rev" "wmean_exp_fast" "wmean_exp_med" "wmean_exp_slow")
SEEDS=(42 43 44 45 46 47 48 49 50 51)  # n=10!

# Hyperparameters
LR=0.001
BS=64
EPOCHS=300
WD=0.0
PATIENCE=60
HIDDEN=256
DROPOUT=0.3

# ============================================================================
# PATHS (EDIT THESE)
# ============================================================================
BASE_DIR="${SENTICCRYSTAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EMB_BASE="${BASE_DIR}/data/embeddings/${TASK}"
RESULTS_BASE="${BASE_DIR}/results_n10/${TASK}/flat"

echo "=================================="
echo "GPU 1: 4way LSTM Baseline (n=10)"
echo "=================================="
echo "Encoders: ${ENCODERS[@]}"
echo "Layers: ${LAYERS[@]}"
echo "Pools: ${POOLS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""

TOTAL=$((${#ENCODERS[@]} * ${#LAYERS[@]} * ${#POOLS[@]} * ${#SEEDS[@]}))
echo "Total runs: ${TOTAL}"
echo ""

SUMMARY_DIR="${BASE_DIR}/results_n10/${TASK}/summary"
mkdir -p "${SUMMARY_DIR}"
SUMMARY_FILE="${SUMMARY_DIR}/gpu1_lstm_summary.txt"
echo "GPU 1: 4way LSTM (n=10)" > "${SUMMARY_FILE}"
echo "Started: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

count=0
success_count=0
fail_count=0
start_time=$(date +%s)

for encoder in "${ENCODERS[@]}"; do
    for layer in "${LAYERS[@]}"; do
        for pool in "${POOLS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                ((count++))
                
                EMB_ROOT="${EMB_BASE}/${encoder}"
                OUT_DIR="${RESULTS_BASE}/${encoder}/${layer}/${pool}/${MODEL}/seed_${seed}"
                
                echo ""
                echo "[$count/$TOTAL] ${encoder}/${layer}/${pool}/${MODEL}/seed_${seed}"
                
                python "${TRAIN_SCRIPT}" \
                    --layer "${layer}" \
                    --pool "${pool}" \
                    --model "${MODEL}" \
                    --emb_root "${EMB_ROOT}" \
                    --out_dir "${OUT_DIR}" \
                    --learning_rate ${LR} \
                    --batch_size ${BS} \
                    --num_epochs ${EPOCHS} \
                    --weight_decay ${WD} \
                    --early_stopping_patience ${PATIENCE} \
                    --hidden_size ${HIDDEN} \
                    --dropout_rate ${DROPOUT} \
                    --seed ${seed}
                
                EXIT_CODE=$?
                
                if [ $EXIT_CODE -eq 0 ]; then
                    ((success_count++))
                    RESULTS_FILE="${OUT_DIR}/results.json"
                    if [ -f "${RESULTS_FILE}" ]; then
                        ACC=$(python3 -c "import json; print(json.load(open('${RESULTS_FILE}'))['metrics']['accuracy'])" 2>/dev/null || echo "N/A")
                        F1W=$(python3 -c "import json; print(json.load(open('${RESULTS_FILE}'))['metrics']['weighted_f1'])" 2>/dev/null || echo "N/A")
                        echo "  ✅ Acc: ${ACC} | F1: ${F1W}"
                        echo "✅ [$count] ${encoder}/${layer}/${pool}/seed_${seed}: Acc=${ACC}, F1=${F1W}" >> "${SUMMARY_FILE}"
                    fi
                else
                    ((fail_count++))
                    echo "  ❌ Failed (exit: ${EXIT_CODE})"
                    echo "❌ [$count] ${encoder}/${layer}/${pool}/seed_${seed}: FAILED" >> "${SUMMARY_FILE}"
                fi
            done
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "=================================="
echo "✅ GPU 1 Complete"
echo "=================================="
echo "Success: ${success_count}/${TOTAL}"
echo "Failed: ${fail_count}/${TOTAL}"
echo "Time: $((total_time / 3600))h $((total_time % 3600 / 60))m"
echo "Summary: ${SUMMARY_FILE}"

echo "" >> "${SUMMARY_FILE}"
echo "Completed: $(date)" >> "${SUMMARY_FILE}"
echo "Success: ${success_count}/${TOTAL}" >> "${SUMMARY_FILE}"
echo "Time: $((total_time / 3600))h $((total_time % 3600 / 60))m" >> "${SUMMARY_FILE}"
