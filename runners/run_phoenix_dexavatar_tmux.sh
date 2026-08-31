#!/bin/bash
# =============================================================================
# Launch DexAvatar Fitting & PA-MPJPE Evaluation on Phoenix in TMUX
# =============================================================================
set -euo pipefail

SESSION_NAME="dexavatar_phoenix_fit"
PROJECT_DIR="/home/haipd/DexAvatar"
PYTHON_BIN="/home/haipd/miniconda3/envs/dexavatar/bin/python"
SPLIT="${1:-dev}"
OUTPUT_DIR="${PROJECT_DIR}/outputs/dexavatar_phoenix_${SPLIT}"
LOG_FILE="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"

echo "======================================================="
echo " Starting DexAvatar Fitting & Evaluation on Phoenix"
echo " Split: ${SPLIT}"
echo " TMUX Session: ${SESSION_NAME}"
echo " Output Dir: ${OUTPUT_DIR}"
echo " Log File: ${LOG_FILE}"
echo "======================================================="

# Check if session already exists
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "[WARN] TMUX session '${SESSION_NAME}' already exists!"
    echo "To attach: tmux attach -t ${SESSION_NAME}"
    echo "To kill:   tmux kill-session -t ${SESSION_NAME}"
    exit 1
fi

# Launch in tmux
tmux new-session -d -s "${SESSION_NAME}" "cd ${PROJECT_DIR} && ${PYTHON_BIN} evaluation/run_phoenix_dexavatar_fitting_eval.py --split ${SPLIT} --output_dir ${OUTPUT_DIR} 2>&1 | tee -a ${LOG_FILE}"

echo "[SUCCESS] Launched DexAvatar Phoenix Evaluation in tmux session '${SESSION_NAME}'"
echo ""
echo "Useful commands:"
echo "  • Attach to session:  tmux attach -t ${SESSION_NAME}"
echo "  • Detach from tmux:   Press Ctrl+B then D"
echo "  • View live log:      tail -f ${LOG_FILE}"
echo "  • Check status:       tmux ls"
echo "  • Kill session:       tmux kill-session -t ${SESSION_NAME}"
echo "======================================================="
