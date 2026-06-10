#!/bin/bash
###############################################################################
# Generate Base Encoder Embeddings - 4way (GPU 0)
###############################################################################

set -e

export CUDA_VISIBLE_DEVICES=0

ENCODERS=(
    "bert-base-uncased:bert-base"
    "roberta-base:roberta-base"
    "sentence-transformers/nli-roberta-base-v2:sentence-roberta"
)

LAYERS=("last" "avg_last4")
POOLS=("mean" "attn" "wmean_pos_rev")

echo "=================================="
echo "4WAY EMBEDDING GENERATION (GPU 0)"
echo "=================================="
echo "Encoders: ${#ENCODERS[@]}"
echo "Layers: ${LAYERS[@]}"
echo "Poolings: ${POOLS[@]}"
echo ""

total=$((${#ENCODERS[@]} * ${#LAYERS[@]}))
count=0

for encoder_spec in "${ENCODERS[@]}"; do
    IFS=':' read -r model_name out_name <<< "$encoder_spec"
    
    for layer in "${LAYERS[@]}"; do
        ((count++))
        echo ""
        echo "[$count/$total] ${out_name} / ${layer}"
        echo "========================================"
        
        python generate_sroberta_npz_4way.py \
            --model_name "${model_name}" \
            --out_root "../data/embeddings/4way/${out_name}" \
            --layer "${layer}" \
            --poolings mean attn wmean_pos_rev \
            --attn_tau 1.0
        
        echo "✅ Completed: ${out_name} / ${layer}"
    done
done

echo ""
echo "=================================="
echo "✅ 4WAY GENERATION COMPLETE"
echo "=================================="