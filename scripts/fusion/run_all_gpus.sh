#!/bin/bash
###############################################################################
# Master Launcher: Run all GPU scripts in parallel
# Total: 3 GPUs × 40 runs = 120 runs
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================================"
echo "SENTICNET EXPERIMENTS - PARALLEL EXECUTION"
echo "================================================================================"
echo "GPU 0: BERT-base-hier           (40 runs)"
echo "GPU 1: RoBERTa-base-hier        (40 runs)"
echo "GPU 2: Sentence-RoBERTa-hier    (40 runs)"
echo "────────────────────────────────────────"
echo "Total:                           120 runs"
echo "================================================================================"
echo ""
echo "Starting all GPUs in parallel..."
echo "Each GPU will run independently in the background."
echo ""

# Make scripts executable
chmod +x "${SCRIPT_DIR}/run_gpu0_bert.sh"
chmod +x "${SCRIPT_DIR}/run_gpu1_roberta.sh"
chmod +x "${SCRIPT_DIR}/run_gpu2_sentence_roberta.sh"

# Launch GPU 0 (BERT)
echo "[GPU 0] Launching BERT experiments..."
nohup "${SCRIPT_DIR}/run_gpu0_bert.sh" > gpu0_bert.out 2>&1 &
PID_GPU0=$!
echo "[GPU 0] PID: $PID_GPU0"

sleep 2

# Launch GPU 1 (RoBERTa)
echo "[GPU 1] Launching RoBERTa experiments..."
nohup "${SCRIPT_DIR}/run_gpu1_roberta.sh" > gpu1_roberta.out 2>&1 &
PID_GPU1=$!
echo "[GPU 1] PID: $PID_GPU1"

sleep 2

# Launch GPU 2 (Sentence-RoBERTa)
echo "[GPU 2] Launching Sentence-RoBERTa experiments..."
nohup "${SCRIPT_DIR}/run_gpu2_sentence_roberta.sh" > gpu2_sentence_roberta.out 2>&1 &
PID_GPU2=$!
echo "[GPU 2] PID: $PID_GPU2"

echo ""
echo "================================================================================"
echo "✅ ALL GPUS LAUNCHED!"
echo "================================================================================"
echo "GPU 0 (BERT):              PID ${PID_GPU0} → gpu0_bert.out"
echo "GPU 1 (RoBERTa):           PID ${PID_GPU1} → gpu1_roberta.out"
echo "GPU 2 (Sentence-RoBERTa):  PID ${PID_GPU2} → gpu2_sentence_roberta.out"
echo "================================================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f gpu0_bert.out"
echo "  tail -f gpu1_roberta.out"
echo "  tail -f gpu2_sentence_roberta.out"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Check running processes:"
echo "  ps aux | grep run_gpu"
echo ""
echo "Kill all (if needed):"
echo "  kill ${PID_GPU0} ${PID_GPU1} ${PID_GPU2}"
echo ""
echo "================================================================================"
echo "Experiments are running in the background. You can safely close this terminal."
echo "================================================================================"

# Save PIDs to file for easy cleanup
echo "${PID_GPU0}" > gpu_pids.txt
echo "${PID_GPU1}" >> gpu_pids.txt
echo "${PID_GPU2}" >> gpu_pids.txt

echo "PIDs saved to gpu_pids.txt"
