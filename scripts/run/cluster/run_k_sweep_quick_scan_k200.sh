#!/bin/bash
# run_k_sweep_quick_scan_k200.sh
# Quick scan to K=200 with step=20

cd "${SENTICCRYSTAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

echo "========================================"
echo "QUICK SCAN: K=0-200 (step=20)"
echo "========================================"
echo "Seed: 42"
echo "Time estimate: ~3-4 hours (2 GPUs)"
echo ""

# 4-way (GPU 0)
CUDA_VISIBLE_DEVICES=0 python train_turnlevel_k_sweep_bayesian.py \
    --task 4way \
    --model_tag sentence-roberta-hier \
    --layer avg_last4 --pool mean \
    --gpu 0 --seed 42 \
    --k_min 0 --k_max 200 --k_step 20 &

# 6-way (GPU 1)
CUDA_VISIBLE_DEVICES=1 python train_turnlevel_k_sweep_bayesian.py \
    --task 6way \
    --model_tag sentence-roberta-hier \
    --layer avg_last4 --pool mean \
    --gpu 1 --seed 42 \
    --k_min 0 --k_max 200 --k_step 20 &

wait

echo ""
echo "✅ Quick scan complete!"
echo ""
echo "Check saturation point:"
echo "  python analyze_saturation.py"
