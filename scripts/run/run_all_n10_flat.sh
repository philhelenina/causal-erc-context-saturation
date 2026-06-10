#!/bin/bash
###############################################################################
# Master Script: Launch all Flat experiments (n=10)
# Total: 720 runs (180 per GPU)
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "🚀 Flat Baseline Experiments (n=10)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Encoders: bert-base, roberta-base, sentence-roberta"
echo "  • Layers: avg_last4, last"
echo "  • Pools: mean, attn, wmean_pos_rev"
echo "  • Classifiers: MLP, LSTM"
echo "  • Tasks: 4way, 6way"
echo "  • Seeds: 42-51 (n=10)"
echo ""
echo "GPU Distribution:"
echo "  • GPU 0: 4way MLP = 180 runs"
echo "  • GPU 1: 4way LSTM = 180 runs"
echo "  • GPU 2: 6way MLP = 180 runs"
echo "  • GPU 3: 6way LSTM = 180 runs"
echo ""
echo "Total: 720 runs"
echo "=========================================="
echo ""

# Make scripts executable
chmod +x "${SCRIPT_DIR}/flat/run_n10_gpu0_4way_mlp.sh"
chmod +x "${SCRIPT_DIR}/flat/run_n10_gpu1_4way_lstm.sh"
chmod +x "${SCRIPT_DIR}/flat/run_n10_gpu2_6way_mlp.sh"
chmod +x "${SCRIPT_DIR}/flat/run_n10_gpu3_6way_lstm.sh"

# Launch all GPUs in parallel
echo "Launching GPU 0 (4way MLP)..."
"${SCRIPT_DIR}/flat/run_n10_gpu0_4way_mlp.sh" > "${SCRIPT_DIR}/n10_gpu0_flat.log" 2>&1 &
PID0=$!

echo "Launching GPU 1 (4way LSTM)..."
"${SCRIPT_DIR}/flat/run_n10_gpu1_4way_lstm.sh" > "${SCRIPT_DIR}/n10_gpu1_flat.log" 2>&1 &
PID1=$!

echo "Launching GPU 2 (6way MLP)..."
"${SCRIPT_DIR}/flat/run_n10_gpu2_6way_mlp.sh" > "${SCRIPT_DIR}/n10_gpu2_flat.log" 2>&1 &
PID2=$!

echo "Launching GPU 3 (6way LSTM)..."
"${SCRIPT_DIR}/flat/run_n10_gpu3_6way_lstm.sh" > "${SCRIPT_DIR}/n10_gpu3_flat.log" 2>&1 &
PID3=$!

echo ""
echo "All GPUs launched! PIDs: $PID0, $PID1, $PID2, $PID3"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu0_flat.log"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu1_flat.log"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu2_flat.log"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu3_flat.log"
echo ""
echo "Waiting for all GPUs to complete..."

# Wait for all background jobs
wait $PID0
EXIT0=$?
wait $PID1
EXIT1=$?
wait $PID2
EXIT2=$?
wait $PID3
EXIT3=$?

echo ""
echo "=========================================="
echo "✅ All Flat Experiments Complete!"
echo "=========================================="
echo "Exit codes: GPU0=$EXIT0, GPU1=$EXIT1, GPU2=$EXIT2, GPU3=$EXIT3"
echo ""
echo "Check logs at:"
echo "  ${SCRIPT_DIR}/n10_gpu0_flat.log"
echo "  ${SCRIPT_DIR}/n10_gpu1_flat.log"
echo "  ${SCRIPT_DIR}/n10_gpu2_flat.log"
echo "  ${SCRIPT_DIR}/n10_gpu3_flat.log"
echo ""
