#!/bin/bash
###############################################################################
# Generate Flat Embeddings for BERT, RoBERTa, Sentence-RoBERTa
# Layers: last, avg_last4
# Poolings: mean, wmean_pos_rev, wmean_exp_fast/med/slow, attn
# Tasks: 4way, 6way
###############################################################################

set -e  # Exit on error

BASE_DIR="${SENTICCRYSTAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${BASE_DIR}"

# Pooling variants (6 total)
POOLINGS="wmean_exp_fast wmean_exp_med wmean_exp_slow"

# Layer variants (2 total)
LAYERS="last avg_last4"

# Tasks
TASKS="4way 6way"

echo "==========================================="
echo "FLAT EMBEDDING GENERATION"
echo "==========================================="
echo "Encoders: BERT, RoBERTa, Sentence-RoBERTa"
echo "Layers: ${LAYERS}"
echo "Poolings: ${POOLINGS}"
echo "Tasks: ${TASKS}"
echo ""

###############################################################################
# 1. BERT-base (bert-base-uncased)
###############################################################################

echo ""
echo "==========================================="
echo "[1/3] BERT-base-uncased"
echo "==========================================="
echo ""

for task in ${TASKS}; do
    for layer in ${LAYERS}; do
        echo ""
        echo ">>> BERT | ${task} | ${layer}"
        
        if [ "$task" == "4way" ]; then
            SCRIPT="scripts/generator/generate_bert_base_npz_4way.py"
            DATA_DIR="data/iemocap_4way_data"
            OUT_ROOT="data/embeddings/4way/bert-base"
        else
            SCRIPT="scripts/generator/generate_bert_base_npz_6way.py"
            DATA_DIR="data/iemocap_6way_data"
            OUT_ROOT="data/embeddings/6way/bert-base"
        fi
        
        python "${SCRIPT}" \
            --model_name "bert-base-uncased" \
            --data_dir "${DATA_DIR}" \
            --out_root "${OUT_ROOT}" \
            --layer "${layer}" \
            --poolings ${POOLINGS} \
            --splits train val test \
            --batch_size 64 \
            --max_length 128
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Success: BERT | ${task} | ${layer}"
        else
            echo "  ❌ Failed: BERT | ${task} | ${layer}"
        fi
    done
done

###############################################################################
# 2. RoBERTa-base
###############################################################################

echo ""
echo "==========================================="
echo "[2/3] RoBERTa-base"
echo "==========================================="
echo ""

for task in ${TASKS}; do
    for layer in ${LAYERS}; do
        echo ""
        echo ">>> RoBERTa | ${task} | ${layer}"
        
        if [ "$task" == "4way" ]; then
            SCRIPT="scripts/generator/generate_roberta_base_npz_4way.py"
            DATA_DIR="data/iemocap_4way_data"
            OUT_ROOT="data/embeddings/4way/roberta-base"
        else
            SCRIPT="scripts/generator/generate_roberta_base_npz_6way.py"
            DATA_DIR="data/iemocap_6way_data"
            OUT_ROOT="data/embeddings/6way/roberta-base"
        fi
        
        python "${SCRIPT}" \
            --model_name "roberta-base" \
            --data_dir "${DATA_DIR}" \
            --out_root "${OUT_ROOT}" \
            --layer "${layer}" \
            --poolings ${POOLINGS} \
            --splits train val test \
            --batch_size 64 \
            --max_length 128
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Success: RoBERTa | ${task} | ${layer}"
        else
            echo "  ❌ Failed: RoBERTa | ${task} | ${layer}"
        fi
    done
done

###############################################################################
# 3. Sentence-RoBERTa (nli-roberta-base-v2)
###############################################################################

echo ""
echo "==========================================="
echo "[3/3] Sentence-RoBERTa"
echo "==========================================="
echo ""

for task in ${TASKS}; do
    for layer in ${LAYERS}; do
        echo ""
        echo ">>> Sentence-RoBERTa | ${task} | ${layer}"
        
        if [ "$task" == "4way" ]; then
            SCRIPT="scripts/generator/generate_sroberta_npz_4way.py"
            DATA_DIR="data/iemocap_4way_data"
            OUT_ROOT="data/embeddings/4way/sentence-roberta"
        else
            SCRIPT="scripts/generator/generate_sroberta_npz_6way.py"
            DATA_DIR="data/iemocap_6way_data"
            OUT_ROOT="data/embeddings/6way/sentence-roberta"
        fi
        
        python "${SCRIPT}" \
            --model_name "sentence-transformers/nli-roberta-base-v2" \
            --data_dir "${DATA_DIR}" \
            --out_root "${OUT_ROOT}" \
            --layer "${layer}" \
            --poolings ${POOLINGS} \
            --splits train val test \
            --batch_size 64 \
            --max_length 128 \
            --attn_tau 1.0 \
            --exp_tau_fast 2.0 \
            --exp_tau_med 5.0 \
            --exp_tau_slow 10.0
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Success: Sentence-RoBERTa | ${task} | ${layer}"
        else
            echo "  ❌ Failed: Sentence-RoBERTa | ${task} | ${layer}"
        fi
    done
done

###############################################################################
# Summary
###############################################################################

echo ""
echo "==========================================="
echo "✅ FLAT EMBEDDING GENERATION COMPLETE"
echo "==========================================="
echo ""
echo "Generated embeddings for:"
echo "  - 3 encoders (BERT, RoBERTa, Sentence-RoBERTa)"
echo "  - 2 layers (last, avg_last4)"
echo "  - 6 poolings (mean, wmean_pos_rev, wmean_exp_fast/med/slow, attn)"
echo "  - 2 tasks (4way, 6way)"
echo ""
echo "Total configs: 3 × 2 × 6 × 2 = 72 embeddings"
echo ""
echo "Output locations:"
echo "  - data/embeddings/4way/{bert-base,roberta-base,sentence-roberta}/{layer}/{pool}/"
echo "  - data/embeddings/6way/{bert-base,roberta-base,sentence-roberta}/{layer}/{pool}/"
echo ""
echo "Next step: Train classifiers on these embeddings"
