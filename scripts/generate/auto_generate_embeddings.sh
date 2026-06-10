#!/bin/bash
# auto_generate_embeddings.sh
# 모든 임베딩을 자동으로 순차 생성

set -e  # 에러 발생시 중단

LOG_FILE="embedding_generation.log"

echo "================================================================================"
echo "AUTOMATIC EMBEDDING GENERATION"
echo "================================================================================"
echo "Start time: $(date)"
echo "Log file: $LOG_FILE"
echo "================================================================================"
echo ""

# Function to run command and log
run_cmd() {
    echo ""
    echo ">>> Running: $@"
    echo "[$(date)] Running: $@" >> $LOG_FILE
    
    if "$@" >> $LOG_FILE 2>&1; then
        echo "✓ Success: $@"
        echo "[$(date)] ✓ Success: $@" >> $LOG_FILE
    else
        echo "✗ Failed: $@"
        echo "[$(date)] ✗ Failed: $@" >> $LOG_FILE
        exit 1
    fi
}

# 1. SenticNet (~5분)
#echo ""
#echo "================================================================================"
#echo "STEP 1/4: SenticNet Generation (~5 minutes)"
#echo "================================================================================"
#run_cmd python generate_senticnet_sentence_level_4way.py
#run_cmd python generate_senticnet_sentence_level_6way.py

# 2. BERT (~15분)
echo ""
echo "================================================================================"
echo "STEP 2/4: BERT Hierarchical Generation (~15 minutes)"
echo "================================================================================"
run_cmd python generate_bert_base_hier_npz_4way.py --layer avg_last4 --poolings mean
run_cmd python generate_bert_base_hier_npz_6way.py --layer avg_last4 --poolings mean

# 3. RoBERTa (~15분)
echo ""
echo "================================================================================"
echo "STEP 3/4: RoBERTa Hierarchical Generation (~15 minutes)"
echo "================================================================================"
run_cmd python generate_roberta_base_hier_npz_4way.py --layer avg_last4 --poolings mean
run_cmd python generate_roberta_base_hier_npz_6way.py --layer avg_last4 --poolings mean

# 4. Sentence-RoBERTa (~15분)
echo ""
echo "================================================================================"
echo "STEP 4/4: Sentence-RoBERTa Hierarchical Generation (~15 minutes)"
echo "================================================================================"
run_cmd python generate_sroberta_hier_npz_4way.py --layer avg_last4 --poolings mean
run_cmd python generate_sroberta_hier_npz_6way.py --layer avg_last4 --poolings mean

# 완료
echo ""
echo "================================================================================"
echo "ALL EMBEDDINGS GENERATED SUCCESSFULLY!"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Verifying embeddings..."
echo ""
