#!/bin/bash
# Quick RunPod setup for ExPhil training
# Run this after SSH-ing into a fresh RunPod instance

set -e

echo "=== ExPhil RunPod Setup ==="

# Install rclone if not present
if ! command -v rclone &> /dev/null; then
    echo "Installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

# Configure B2 (you'll need to enter your credentials)
if ! rclone listremotes | grep -q "^b2:"; then
    echo ""
    echo "Configure Backblaze B2:"
    echo "  1. Select 'b2' as remote type"
    echo "  2. Enter your B2 Application Key ID"
    echo "  3. Enter your B2 Application Key"
    echo ""
    rclone config
fi

# Clone or update exphil
if [[ -d "/workspace/exphil" ]]; then
    echo "Updating exphil..."
    cd /workspace/exphil
    git pull
else
    echo "Cloning exphil..."
    cd /workspace
    git clone https://github.com/YOUR_USERNAME/exphil.git  # Update this URL
    cd exphil
fi

# Install Elixir deps
echo "Installing dependencies..."
mix local.hex --force
mix local.rebar --force
mix deps.get

# Download replays from B2
echo "Downloading Mewtwo replays..."
mkdir -p /workspace/replays
rclone copy "b2:${B2_REPLAYS:-your-replays-bucket}/mewtwo/" /workspace/replays/mewtwo/ --progress

# Create checkpoint dir
mkdir -p /workspace/checkpoints
mkdir -p /workspace/logs

# Link checkpoint dirs to exphil
ln -sf /workspace/checkpoints /workspace/exphil/checkpoints
ln -sf /workspace/logs /workspace/exphil/logs

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick commands:"
echo "  # Train a model"
echo "  cd /workspace/exphil"
echo "  mix run scripts/train_from_replays.exs --replays /workspace/replays/mewtwo --train-character mewtwo --temporal --backbone mamba --epochs 10 --name my_model"
echo ""
echo "  # Push results to B2"
echo "  ./scripts/sync_runpod.sh push"
echo ""
