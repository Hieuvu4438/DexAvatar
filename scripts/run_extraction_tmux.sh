#!/bin/bash
# Run PHOENIX14T extraction in tmux session
# Tạo tmux session để tránh bị ngắt khi server disconnect
#
# Usage:
#   bash scripts/run_extraction_tmux.sh
#
# Kiểm tra progress:
#   tmux attach -t phoenix_extract
#
# Detach: Ctrl+B, D

set -e

SESSION_NAME="phoenix_extract"
PHOENIX_DIR="/home/haipd/DexAvatar/data/signbposer_data/raw/phoenix/phoenix-2014-T"
OUTPUT_DIR="/home/haipd/DexAvatar/data/signbposer_data/raw/phoenix"
SCRIPT_DIR="/home/haipd/DexAvatar/scripts"

echo "============================================"
echo "PHOENIX14T Extraction - TMUX Session"
echo "============================================"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux not found. Installing..."
    sudo apt-get install -y tmux
fi

# Kill existing session if exists
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create new tmux session
echo "Creating tmux session: $SESSION_NAME"
tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Send commands to tmux session
tmux send-keys -t "$SESSION_NAME" "cd /home/haipd/DexAvatar" Enter
tmux send-keys -t "$SESSION_NAME" "echo 'Starting PHOENIX14T extraction...'" Enter
tmux send-keys -t "$SESSION_NAME" "conda run -n smpler_x python3 scripts/extract_phoenix14t_body_pose.py \
    --phoenix_dir $PHOENIX_DIR \
    --output_dir $OUTPUT_DIR \
    --temp_dir /tmp/phoenix_extract_frames \
    --max_frames_per_video 10 \
    --gpu_id 0 \
    2>&1 | tee /home/haipd/DexAvatar/data/signbposer_data/raw/phoenix/extraction.log" Enter

echo ""
echo "Tmux session created!"
echo ""
echo "Commands:"
echo "  Attach to session:  tmux attach -t $SESSION_NAME"
echo "  Detach from session: Ctrl+B, then D"
echo "  Check progress:     tail -f $OUTPUT_DIR/extraction.log"
echo "  Kill session:       tmux kill-session -t $SESSION_NAME"
echo ""
echo "Extraction is running in background..."
