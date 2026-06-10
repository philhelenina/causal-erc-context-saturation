#!/bin/bash
###############################################################################
# Master Script: Launch all Hierarchical experiments (n=10)
# Total: 480 runs (120 per GPU)
###############################################################################

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=========================================="
echo "🚀 Hierarchical Experiments (n=10)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Encoder: sentence-roberta-hier"
echo "  • Layers: avg_last4, last"
echo "  • Pools: mean, wmean_pos_rev"
echo "  • Aggregators: mean, attn, sum, expdecay, lstm"
echo "  • Classifiers: MLP, LSTM"
echo "  • Tasks: 4way, 6way"
echo "  • Seeds: 42-51 (n=10)"
echo ""
echo "GPU Distribution:"
echo "  • GPU 0: 4way (mean, lstm, attn-mlp) = 120 runs"
echo "  • GPU 1: 4way (attn-lstm, expdecay, sum) = 120 runs"
echo "  • GPU 2: 6way (mean, lstm, attn-mlp) = 120 runs"
echo "  • GPU 3: 6way (attn-lstm, expdecay, sum) = 120 runs"
echo ""
echo "Total: 480 runs"
echo "=========================================="
echo ""

# Make scripts executable
chmod +x "${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu0_4way.sh"
chmod +x "${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu1_4way.sh"
chmod +x "${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu2_6way.sh"
chmod +x "${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu3_6way.sh"

# Launch all GPUs in parallel
echo "Launching GPU 0 (4way hierarchical)..."
"${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu0_4way.sh" > "${SCRIPT_DIR}/n10_gpu0_hier.log" 2>&1 &
PID0=$!

echo "Launching GPU 1 (4way hierarchical)..."
"${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu1_4way.sh" > "${SCRIPT_DIR}/n10_gpu1_hier.log" 2>&1 &
PID1=$!

echo "Launching GPU 2 (6way hierarchical)..."
"${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu2_6way.sh" > "${SCRIPT_DIR}/n10_gpu2_hier.log" 2>&1 &
PID2=$!

echo "Launching GPU 3 (6way hierarchical)..."
"${SCRIPT_DIR}/hierarchical/run_n10_hier_gpu3_6way.sh" > "${SCRIPT_DIR}/n10_gpu3_hier.log" 2>&1 &
PID3=$!

echo ""
echo "All GPUs launched! PIDs: $PID0, $PID1, $PID2, $PID3"
echo ""
echo "Monitor progress with:"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu0_hier.log"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu1_hier.log"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu2_hier.log"
echo "  tail -f ${SCRIPT_DIR}/n10_gpu3_hier.log"
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
echo "✅ All Hierarchical Experiments Complete!"
echo "=========================================="
echo "Exit codes: GPU0=$EXIT0, GPU1=$EXIT1, GPU2=$EXIT2, GPU3=$EXIT3"
echo ""
echo "Check logs at:"
echo "  ${SCRIPT_DIR}/n10_gpu0_hier.log"
echo "  ${SCRIPT_DIR}/n10_gpu1_hier.log"
echo "  ${SCRIPT_DIR}/n10_gpu2_hier.log"
echo "  ${SCRIPT_DIR}/n10_gpu3_hier.log"
echo ""
