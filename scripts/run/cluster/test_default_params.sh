#!/bin/bash
# Test run: mean pooling with DEFAULT params (vs Bayesian)
#
# Purpose: Check if default params (lr=0.001, dropout=0.3, epochs=60)
#          perform similarly to Bayesian params (lr=0.0002, dropout=0.49, epochs=119)
#
# Test seed: 52 (new seed for testing)
# Run on: GPU server (Saturn Cloud or your GPU machine)

echo "======================================"
echo "TEST RUN: DEFAULT vs BAYESIAN PARAMS"
echo "======================================"
echo "Pool:        mean"
echo "Seed:        52 (test)"
echo "Params:      DEFAULT (lr=0.001, dropout=0.3, epochs=60, wd=0.0)"
echo "Compare to:  seed42-48 (Bayesian params)"
echo "======================================"
echo "Start time: $(date)"
echo ""

# Configuration
SCRIPT="scripts/train/turn/train_turnlevel_k_sweep_main.py"
MODEL_TAG="sentence-roberta-hier"
LAYER="avg_last4"
POOL="mean"
AGGREGATOR="mean"
TASK="4way"
SEED=52

# GPU
GPU="0"

# Log directory
mkdir -p test_logs

# MANUAL HYPERPARAMETERS (force default, ignore Bayesian file)
LR=0.001
HIDDEN=256
DROPOUT=0.3
BS=64
EPOCHS=60
WD=0.0

echo "Hyperparameters (MANUAL - DEFAULT):"
echo "  Learning rate:  ${LR}"
echo "  Hidden size:    ${HIDDEN}"
echo "  Dropout:        ${DROPOUT}"
echo "  Batch size:     ${BS}"
echo "  Epochs:         ${EPOCHS}"
echo "  Weight decay:   ${WD}"
echo ""

LOG_FILE="test_logs/test_mean_default_seed${SEED}.log"

echo "Running experiment..."
echo "Log: ${LOG_FILE}"
echo ""

# Run with MANUAL params (will override Bayesian)
python "${SCRIPT}" \
    --task "${TASK}" \
    --model_tag "${MODEL_TAG}" \
    --layer "${LAYER}" \
    --pool "${POOL}" \
    --aggregator "${AGGREGATOR}" \
    --seed "${SEED}" \
    --k_min 0 --k_max 200 --k_step 10 \
    --lr "${LR}" \
    --hidden_size "${HIDDEN}" \
    --dropout "${DROPOUT}" \
    --bs "${BS}" \
    --epochs "${EPOCHS}" \
    --weight_decay "${WD}" \
    --gpu "${GPU}" \
    > "${LOG_FILE}" 2>&1

EXIT_CODE=$?

echo ""
echo "======================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ TEST COMPLETED SUCCESSFULLY"
else
    echo "✗ TEST FAILED (exit code: ${EXIT_CODE})"
fi
echo "======================================"
echo "End time: $(date)"
echo ""

# Show result location
RESULT_DIR="../../results/turnlevel_k_sweep_bayesian/4way_sentence-roberta-hier_avg_last4_mean_mean/seed${SEED}"

if [ -f "${RESULT_DIR}/metadata.json" ]; then
    echo "Result saved to:"
    echo "  ${RESULT_DIR}/"
    echo ""
    echo "Metadata:"
    cat "${RESULT_DIR}/metadata.json"
    echo ""
    echo ""
    echo "Performance (last 5 K values):"
    if [ -f "${RESULT_DIR}/k_sweep_results.csv" ]; then
        tail -6 "${RESULT_DIR}/k_sweep_results.csv"
    fi
    echo ""
    echo "Best performance:"
    if [ -f "${RESULT_DIR}/k_sweep_results.csv" ]; then
        python3 << 'PYEOF'
import pandas as pd
import sys

try:
    df = pd.read_csv('../../results/turnlevel_k_sweep_bayesian/4way_sentence-roberta-hier_avg_last4_mean_mean/seed52/k_sweep_results.csv')
    best_idx = df['f1_weighted'].idxmax()
    best = df.loc[best_idx]
    print(f"  K={int(best['K'])}: F1={best['f1_weighted']:.4f}, Acc={best['accuracy']:.4f}")
except:
    print("  (Could not parse CSV)")
PYEOF
    fi
else
    echo "Result directory not found. Check log:"
    echo "  ${LOG_FILE}"
fi

echo ""
echo "======================================"
echo "COMPARISON WITH BAYESIAN PARAMS:"
echo "======================================"

# Show Bayesian results for comparison
BAYESIAN_SEEDS=(42 43 44 45 46 47 48)
echo ""
echo "Bayesian params results (seed42-48):"

for s in "${BAYESIAN_SEEDS[@]}"; do
    BAYESIAN_DIR="../../results/turnlevel_k_sweep_bayesian/4way_sentence-roberta-hier_avg_last4_mean_mean/seed${s}"
    if [ -f "${BAYESIAN_DIR}/k_sweep_results.csv" ]; then
        python3 << PYEOF
import pandas as pd
try:
    df = pd.read_csv('${BAYESIAN_DIR}/k_sweep_results.csv')
    best_idx = df['f1_weighted'].idxmax()
    best = df.loc[best_idx]
    print(f"  seed${s}: K={int(best['K'])}, F1={best['f1_weighted']:.4f}, Acc={best['accuracy']:.4f}")
except:
    pass
PYEOF
    fi
done

echo ""
echo "Default params result (seed52):"
if [ -f "${RESULT_DIR}/k_sweep_results.csv" ]; then
    python3 << 'PYEOF'
import pandas as pd
try:
    df = pd.read_csv('../../results/turnlevel_k_sweep_bayesian/4way_sentence-roberta-hier_avg_last4_mean_mean/seed52/k_sweep_results.csv')
    best_idx = df['f1_weighted'].idxmax()
    best = df.loc[best_idx]
    print(f"  seed52: K={int(best['K'])}, F1={best['f1_weighted']:.4f}, Acc={best['accuracy']:.4f}")
except:
    print("  (Results not available)")
PYEOF
fi

echo ""
echo "======================================"
echo "NEXT STEPS:"
echo "======================================"
echo "1. Compare F1 scores:"
echo "   - If seed52 (default) ≈ seed42-48 (Bayesian): Use default for all"
echo "   - If seed52 << seed42-48: Need to use Bayesian for fair comparison"
echo ""
echo "2. Decision:"
echo "   → If default is OK: Run all with default (fair comparison)"
echo "   → If default is bad: Run Bayesian opt for wmean_pos/wmean_pos_rev"
echo ""
echo "======================================"
