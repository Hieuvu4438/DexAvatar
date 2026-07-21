#!/bin/bash

# Create data directories if they do not exist
mkdir -p /home/haipd/DexAvatar/data/InterHand2.6M
mkdir -p /home/haipd/DexAvatar/data/ARCTIC
mkdir -p /home/haipd/DexAvatar/data/WHIM

# Start a new tmux session named 'dataset_downloads'
# The first window (0) is for InterHand2.6M
tmux new-session -d -s dataset_downloads -n "InterHand2.6M"

# Setup Window 0 (InterHand2.6M)
tmux send-keys -t dataset_downloads:0 "cd /home/haipd/DexAvatar/data/InterHand2.6M" C-m
tmux send-keys -t dataset_downloads:0 "echo '=== Downloading InterHand2.6M Annotations (5 fps v1.0) ==='" C-m
tmux send-keys -t dataset_downloads:0 "gdown --folder https://drive.google.com/drive/folders/12RNG9slv9i_TsXSoZ6pQAq-Fa98eGLoy?usp=sharing" C-m

# Setup Window 1 (ARCTIC)
tmux new-window -t dataset_downloads -n "ARCTIC"
tmux send-keys -t dataset_downloads:1 "cd /home/haipd/DexAvatar/data/ARCTIC" C-m
tmux send-keys -t dataset_downloads:1 "echo '=== Downloading ARCTIC Parameters & Set Up Body Models ==='" C-m
tmux send-keys -t dataset_downloads:1 "python3 download_arctic.py" C-m

# Setup Window 2 (WHIM)
tmux new-window -t dataset_downloads -n "WHIM"
tmux send-keys -t dataset_downloads:2 "cd /home/haipd/DexAvatar/data/WHIM" C-m
tmux send-keys -t dataset_downloads:2 "echo '=== Downloading WHIM Annotations ==='" C-m
tmux send-keys -t dataset_downloads:2 "huggingface-cli download --repo-type dataset rolpotamias/WHIM --local-dir /home/haipd/DexAvatar/data/WHIM --local-dir-use-symlinks False" C-m
tmux send-keys -t dataset_downloads:2 "echo '=== Merging and Extracting WHIM splits ==='" C-m
tmux send-keys -t dataset_downloads:2 "cat train_split.z01 train_split.z02 train_split.z03 train_split.zip > train_split_merged.zip && unzip train_split_merged.zip && rm train_split_merged.zip && echo 'WHIM Setup Complete!'" C-m

# Select the first window
tmux select-window -t dataset_downloads:0

echo "Tmux session 'dataset_downloads' created successfully!"
echo "Switch windows inside tmux using: Ctrl+b then 0, 1, or 2."
echo "View progress using: tmux attach -t dataset_downloads"
