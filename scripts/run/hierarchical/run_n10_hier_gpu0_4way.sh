#!/bin/bash
###############################################################################
# Hierarchical GPU 0: 4way (mean, lstm, attn-mlp aggregators)
# 2 layers × 2 pools × 3 agg-clf combos × 10 seeds = 120 runs
###############################################################################

export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAIN_SCRIPT="${BASE_DIR}/scripts/train/train_npz_hier_classifier_4way.py"

# ============================================================================
# CONFIGURATION
# ============================================================================
TASK="4way"
ENCODER="sentence-roberta-hier"
LAYERS=("avg_last4" "last")
POOLS=("mean" "wmean_pos_rev" "wmean_exp_fast" "wmean_exp_med" "wmean_exp_slow")
SEEDS=(42 43 44 45 46 47 48 49 50 51)  # n=10!

# Hyperparameters
LR=0.001
BS=64
EPOCHS=300
WD=0.0
PATIENCE=60
HIDDEN=256
DROPOUT=0.3
DECAY_LAMBDA=0.5

# ============================================================================
# PATHS (EDIT THESE)
# ============================================================================
BASE_DIR="${SENTICCRYSTAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
EMB_BASE="${BASE_DIR}/data/embeddings/${TASK}"
RESULTS_BASE="${BASE_DIR}/results_n10/${TASK}/hierarchical"

echo "==========================================="
echo "Hierarchical GPU 0: 4way (n=10)"
echo "==========================================="
echo "Aggregators: mean(mlp/lstm), lstm(mlp/lstm), attn(mlp only)"
echo "Layers: ${LAYERS[@]}"
echo "Pools: ${POOLS[@]}"
echo "Seeds: ${SEEDS[@]}"
echo ""

count=0
success_count=0
fail_count=0
start_time=$(date +%s)

SUMMARY_DIR="${BASE_DIR}/results_n10/${TASK}/summary"
mkdir -p "${SUMMARY_DIR}"
SUMMARY_FILE="${SUMMARY_DIR}/hier_gpu0_4way_summary.txt"
echo "Hierarchical GPU 0: 4way (n=10)" > "${SUMMARY_FILE}"
echo "Started: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

for layer in "${LAYERS[@]}"; do
    for pool in "${POOLS[@]}"; do
        # mean aggregator: both classifiers
        for clf in "mlp" "lstm"; do
            agg="mean"
            for seed in "${SEEDS[@]}"; do
                ((count++))
                
                EMB_ROOT="${EMB_BASE}/${ENCODER}"
                OUT_DIR="${RESULTS_BASE}/${ENCODER}/${layer}/${pool}/${agg}/${clf}/seed_${seed}"
                
                echo ""
                echo "[$count/120] ${layer}/${pool}/${agg}/${clf}/seed_${seed}"
                
                python "${TRAIN_SCRIPT}" \
                    --layer "${layer}" \
                    --pool "${pool}" \
                    --aggregator "${agg}" \
                    --classifier "${clf}" \
                    --emb_root "${EMB_ROOT}" \
                    --out_dir "${OUT_DIR}" \
                    --lr ${LR} \
                    --batch_size ${BS} \
                    --epochs ${EPOCHS} \
                    --weight_decay ${WD} \
                    --patience ${PATIENCE} \
                    --hidden_size ${HIDDEN} \
                    --dropout ${DROPOUT} \
                    --decay_lambda ${DECAY_LAMBDA} \
                    --decay_reverse \
                    --seed ${seed}
                
                if [ $? -eq 0 ]; then
                    ((success_count++))
                    echo "  ✅ Success"
                    echo "✅ [$count] ${layer}/${pool}/${agg}/${clf}/seed_${seed}" >> "${SUMMARY_FILE}"
                else
                    ((fail_count++))
                    echo "  ❌ Failed"
                    echo "❌ [$count] ${layer}/${pool}/${agg}/${clf}/seed_${seed}" >> "${SUMMARY_FILE}"
                fi
            done
        done
        
        # lstm aggregator: both classifiers
        for clf in "mlp" "lstm"; do
            agg="lstm"
            for seed in "${SEEDS[@]}"; do
                ((count++))
                
                EMB_ROOT="${EMB_BASE}/${ENCODER}"
                OUT_DIR="${RESULTS_BASE}/${ENCODER}/${layer}/${pool}/${agg}/${clf}/seed_${seed}"
                
                echo ""
                echo "[$count/120] ${layer}/${pool}/${agg}/${clf}/seed_${seed}"
                
                python "${TRAIN_SCRIPT}" \
                    --layer "${layer}" \
                    --pool "${pool}" \
                    --aggregator "${agg}" \
                    --classifier "${clf}" \
                    --emb_root "${EMB_ROOT}" \
                    --out_dir "${OUT_DIR}" \
                    --lr ${LR} \
                    --batch_size ${BS} \
                    --epochs ${EPOCHS} \
                    --weight_decay ${WD} \
                    --patience ${PATIENCE} \
                    --hidden_size ${HIDDEN} \
                    --dropout ${DROPOUT} \
                    --decay_lambda ${DECAY_LAMBDA} \
                    --decay_reverse \
                    --seed ${seed}
                
                if [ $? -eq 0 ]; then
                    ((success_count++))
                    echo "  ✅ Success"
                    echo "✅ [$count] ${layer}/${pool}/${agg}/${clf}/seed_${seed}" >> "${SUMMARY_FILE}"
                else
                    ((fail_count++))
                    echo "  ❌ Failed"
                    echo "❌ [$count] ${layer}/${pool}/${agg}/${clf}/seed_${seed}" >> "${SUMMARY_FILE}"
                fi
            done
        done
        
        # attn aggregator: mlp only (lstm will run on GPU1)
        agg="attn"
        clf="mlp"
        for seed in "${SEEDS[@]}"; do
            ((count++))
            
            EMB_ROOT="${EMB_BASE}/${ENCODER}"
            OUT_DIR="${RESULTS_BASE}/${ENCODER}/${layer}/${pool}/${agg}/${clf}/seed_${seed}"
            
            echo ""
            echo "[$count/120] ${layer}/${pool}/${agg}/${clf}/seed_${seed}"
            
            python "${TRAIN_SCRIPT}" \
                --layer "${layer}" \
                --pool "${pool}" \
                --aggregator "${agg}" \
                --classifier "${clf}" \
                --emb_root "${EMB_ROOT}" \
                --out_dir "${OUT_DIR}" \
                --lr ${LR} \
                --batch_size ${BS} \
                --epochs ${EPOCHS} \
                --weight_decay ${WD} \
                --patience ${PATIENCE} \
                --hidden_size ${HIDDEN} \
                --dropout ${DROPOUT} \
                --decay_lambda ${DECAY_LAMBDA} \
                --decay_reverse \
                --seed ${seed}
            
            if [ $? -eq 0 ]; then
                ((success_count++))
                echo "  ✅ Success"
                echo "✅ [$count] ${layer}/${pool}/${agg}/${clf}/seed_${seed}" >> "${SUMMARY_FILE}"
            else
                ((fail_count++))
                echo "  ❌ Failed"
                echo "❌ [$count] ${layer}/${pool}/${agg}/${clf}/seed_${seed}" >> "${SUMMARY_FILE}"
            fi
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo ""
echo "==========================================="
echo "✅ Hierarchical GPU 0 Complete (4way)"
echo "==========================================="
echo "Completed: ${count} runs"
echo "Success: ${success_count}"
echo "Failed: ${fail_count}"
echo "Time: $((total_time / 3600))h $((total_time % 3600 / 60))m"

echo "" >> "${SUMMARY_FILE}"
echo "Completed: $(date)" >> "${SUMMARY_FILE}"
echo "Success: ${success_count}/${count}" >> "${SUMMARY_FILE}"
echo "Time: $((total_time / 3600))h $((total_time % 3600 / 60))m" >> "${SUMMARY_FILE}"
