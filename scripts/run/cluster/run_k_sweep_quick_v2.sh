#!/bin/bash
# run_k_sweep_quick_v2.sh
# Quick test with masking support (v2)

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "QUICK K SWEEP TEST (seed 42) - v2"
echo "========================================"
echo "Working directory: $SCRIPT_DIR"
echo "NEW: Pack_padded_sequence masking!"
echo "Time estimate: ~2-3 hours (2 GPUs)"
echo ""

# 4-way (GPU 0)
CUDA_VISIBLE_DEVICES=0 python train_turnlevel_k_sweep_bayesian_v2.py \
    --task 4way \
    --model_tag sentence-roberta-hier \
    --layer avg_last4 --pool mean \
    --gpu 0 --seed 42 \
    --k_min 0 --k_max 100 --k_step 10 &

# 6-way (GPU 1)
CUDA_VISIBLE_DEVICES=1 python train_turnlevel_k_sweep_bayesian_v2.py \
    --task 6way \
    --model_tag sentence-roberta-hier \
    --layer avg_last4 --pool mean \
    --gpu 1 --seed 42 \
    --k_min 0 --k_max 100 --k_step 10 &

wait

echo ""
echo "========================================"
echo "✅ Quick test complete!"
echo "========================================"
echo ""
echo "Check results:"
echo "  ../../results/turnlevel_k_sweep_bayesian/4way_*/seed42/k_sweep_results.csv"
echo "  ../../results/turnlevel_k_sweep_bayesian/6way_*/seed42/k_sweep_results.csv"
echo ""
echo "Expected improvement:"
echo "  Before (no masking): K=100 → ~66%"
echo "  After (with masking): K=100 → ~69-70% (+3-4%p)"
