#!/bin/bash
###############################################################################
# Minimal Baseline for SenticNet Comparison
# BERT + RoBERTa hierarchical: avg_last4 / mean / mean / mlp only
# 2 encoders × 10 seeds = 20 runs (~1 hour)
###############################################################################

export CUDA_VISIBLE_DEVICES=0

ENCODERS=("bert-base-hier" "roberta-base-hier")
TASK="6way"
LAYER="avg_last4"
POOL="mean"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

echo "==========================================="
echo "Minimal Baseline: BERT + RoBERTa"
echo "==========================================="
echo "Config: avg_last4 / mean / mean / mlp"
echo "Seeds: ${SEEDS[@]}"
echo ""

count=0
for encoder in "${ENCODERS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ((count++))
        echo "[$count/20] ${encoder} - seed ${seed}"
        
        python train_npz_hier_classifier_4way.py \
            --task ${TASK} \
            --encoder ${encoder} \
            --layer ${LAYER} \
            --pool ${POOL} \
            --classifier mlp \
            --seed ${seed}
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Success"
        else
            echo "  ❌ Failed"
        fi
    done
done

echo ""
echo "✅ Baseline complete: ${count} runs"
