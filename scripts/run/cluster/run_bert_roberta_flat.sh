#!/bin/bash
###############################################################################
# Master Launcher: Run BERT + RoBERTa flat experiments on 4 GPUs
# Total: 480 runs (120 per GPU)
###############################################################################

BASE_DIR="${SENTICCRYSTAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SCRIPT_DIR="${BASE_DIR}/scripts/trainer"

echo "==========================================="
echo "BERT + ROBERTA FLAT BASELINE EXPERIMENTS"
echo "==========================================="
echo ""
echo "Configuration:"
echo "  Encoders: bert-base, roberta-base"
echo "  Tasks: 4way, 6way"
echo "  Models: MLP, LSTM"
echo "  Layers: avg_last4, last"
echo "  Pools: wmean_exp_fast, wmean_exp_med, wmean_exp_slow"
echo "  Seeds: 42-51 (n=10)"
echo ""
echo "GPU Distribution:"
echo "  GPU 0: 4way MLP  (120 runs)"
echo "  GPU 1: 4way LSTM (120 runs)"
echo "  GPU 2: 6way MLP  (120 runs)"
echo "  GPU 3: 6way LSTM (120 runs)"
echo ""
echo "Total: 480 runs"
echo ""
echo "==========================================="
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. CUDA not available?"
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Create log directory
LOG_DIR="${BASE_DIR}/logs/bert_roberta_flat"
mkdir -p "${LOG_DIR}"

echo "Starting experiments..."
echo "Logs will be saved to: ${LOG_DIR}/"
echo ""

# Launch all 4 GPUs in parallel
nohup bash "${SCRIPT_DIR}/gpu0_4way_mlp.sh" > "${LOG_DIR}/gpu0_4way_mlp.log" 2>&1 &
PID0=$!
echo "✅ GPU 0 (4way MLP):  PID ${PID0} → ${LOG_DIR}/gpu0_4way_mlp.log"

nohup bash "${SCRIPT_DIR}/gpu1_4way_lstm.sh" > "${LOG_DIR}/gpu1_4way_lstm.log" 2>&1 &
PID1=$!
echo "✅ GPU 1 (4way LSTM): PID ${PID1} → ${LOG_DIR}/gpu1_4way_lstm.log"

nohup bash "${SCRIPT_DIR}/gpu2_6way_mlp.sh" > "${LOG_DIR}/gpu2_6way_mlp.log" 2>&1 &
PID2=$!
echo "✅ GPU 2 (6way MLP):  PID ${PID2} → ${LOG_DIR}/gpu2_6way_mlp.log"

nohup bash "${SCRIPT_DIR}/gpu3_6way_lstm.sh" > "${LOG_DIR}/gpu3_6way_lstm.log" 2>&1 &
PID3=$!
echo "✅ GPU 3 (6way LSTM): PID ${PID3} → ${LOG_DIR}/gpu3_6way_lstm.log"

echo ""
echo "==========================================="
echo "All GPUs launched!"
echo "==========================================="
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/gpu0_4way_mlp.log"
echo "  tail -f ${LOG_DIR}/gpu1_4way_lstm.log"
echo "  tail -f ${LOG_DIR}/gpu2_6way_mlp.log"
echo "  tail -f ${LOG_DIR}/gpu3_6way_lstm.log"
echo ""
echo "Check all at once:"
echo "  watch -n 10 'tail -5 ${LOG_DIR}/gpu*.log'"
echo ""
echo "PIDs: ${PID0} ${PID1} ${PID2} ${PID3}"
echo "To stop all: kill ${PID0} ${PID1} ${PID2} ${PID3}"
echo ""

# Save PID file for easy management
PID_FILE="${LOG_DIR}/pids.txt"
echo "${PID0} ${PID1} ${PID2} ${PID3}" > "${PID_FILE}"
echo "PIDs saved to: ${PID_FILE}"
echo ""

echo "Waiting for completion..."
echo "(This will take several hours)"
echo ""

# Wait for all processes
wait ${PID0} ${PID1} ${PID2} ${PID3}

echo ""
echo "==========================================="
echo "✅ ALL EXPERIMENTS COMPLETE"
echo "==========================================="
echo ""
echo "Check results:"
echo "  - Summary files: ${BASE_DIR}/results_n10/*/summary/"
echo "  - Full logs: ${LOG_DIR}/"
echo ""
