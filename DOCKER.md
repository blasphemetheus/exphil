# ExPhil Cloud GPU Training

Deploy ExPhil to cloud GPU services for faster training.

## Quick Start

```bash
# Build the image locally
docker build -f Dockerfile.gpu -t exphil:gpu .

# Test locally (if you have NVIDIA GPU)
docker-compose run train-gpu
```

## Cloud Deployment

### RunPod (Recommended)

1. **Create account** at [runpod.io](https://runpod.io)

2. **Push image to Docker Hub** (or use RunPod's registry):
   ```bash
   docker tag exphil:gpu yourusername/exphil:gpu
   docker push yourusername/exphil:gpu
   ```

3. **Create a Pod**:
   - Template: Custom Docker Image
   - Image: `yourusername/exphil:gpu`
   - GPU: RTX 4090 ($0.44/hr) or A100 ($1.64/hr)
   - Disk: 50GB+ (for replays)

4. **Upload replays**:
   ```bash
   # From your local machine
   rsync -avz ./replays/ root@<pod-ip>:/app/replays/
   ```

5. **Run training**:
   ```bash
   ssh root@<pod-ip>
   cd /app
   mix run scripts/train_from_replays.exs \
     --epochs 10 \
     --batch-size 128 \
     --replays /app/replays
   ```

6. **Download checkpoints**:
   ```bash
   rsync -avz root@<pod-ip>:/app/checkpoints/ ./checkpoints/
   ```

### Lambda Labs

1. **Create instance** at [lambdalabs.com](https://lambdalabs.com)
   - Choose GPU: A10 ($0.60/hr) or A100 ($1.10/hr)
   - OS: Ubuntu 22.04

2. **SSH and setup**:
   ```bash
   ssh ubuntu@<instance-ip>

   # Install Docker with NVIDIA support
   curl -fsSL https://get.docker.com | sh
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

   # Pull and run
   docker pull yourusername/exphil:gpu
   ```

3. **Run with mounted volumes**:
   ```bash
   docker run --gpus all \
     -v ~/replays:/app/replays:ro \
     -v ~/checkpoints:/app/checkpoints:rw \
     yourusername/exphil:gpu \
     mix run scripts/train_from_replays.exs --epochs 10
   ```

### Vast.ai (Budget Option)

1. **Create account** at [vast.ai](https://vast.ai)

2. **Search for instances**:
   - Filter: RTX 3090/4090, Docker available
   - Sort by: $/hr

3. **Rent and connect**:
   ```bash
   # Use their SSH connection string
   ssh -p <port> root@<host>
   ```

4. **Pull image and run** (same as Lambda Labs)

## Workflow: Local Dev → Cloud Training

```
┌─────────────────────────────────────────────────────────────┐
│  LOCAL DEVELOPMENT                                          │
│  ┌─────────────────┐    ┌─────────────────────────────────┐│
│  │ Edit code       │───→│ Test on 1 replay (CPU)          ││
│  │ lib/*.ex        │    │ mix run scripts/train... --max-1││
│  └─────────────────┘    └─────────────────────────────────┘│
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                       │
│  │ git commit/push │                                       │
│  └────────┬────────┘                                       │
└───────────┼─────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│  CLOUD GPU                                                    │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │ git pull        │───→│ docker build                    │  │
│  │ (or rebuild img)│    │ docker-compose run train-gpu    │  │
│  └─────────────────┘    └─────────────────────────────────┘  │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Full training: 100+ replays, 10+ epochs                 │ │
│  │ ~10-30 min on A100 vs ~4 hours on CPU                   │ │
│  └─────────────────────────────────────────────────────────┘ │
│           │                                                   │
│           ▼                                                   │
│  ┌─────────────────┐                                         │
│  │ Download:       │                                         │
│  │ checkpoints/    │                                         │
│  └────────┬────────┘                                         │
└───────────┼───────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────┐
│  LOCAL EVALUATION                                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ mix run scripts/eval_model.exs --policy checkpoint.bin  │ │
│  │ mix run scripts/play_dolphin.exs --policy checkpoint.bin│ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

## Training Commands

```bash
# Basic training (single-frame MLP)
mix run scripts/train_from_replays.exs \
  --epochs 10 \
  --batch-size 128 \
  --replays /app/replays

# Temporal training (sliding window attention)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone sliding_window \
  --window-size 60 \
  --epochs 10 \
  --batch-size 64

# With Wandb logging
WANDB_API_KEY=your_key mix run scripts/train_from_replays.exs \
  --wandb \
  --wandb-project exphil \
  --epochs 10
```

## Cost Estimates

| GPU | Provider | $/hr | 100 replays, 10 epochs |
|-----|----------|------|------------------------|
| RTX 4090 | RunPod | $0.44 | ~$0.30 (~40 min) |
| A10 | Lambda | $0.60 | ~$0.50 (~50 min) |
| A100 | RunPod | $1.64 | ~$0.55 (~20 min) |
| A100 | Lambda | $1.10 | ~$0.37 (~20 min) |

*Estimates based on ~10x GPU speedup vs CPU for this workload*

## Troubleshooting

### CUDA out of memory
Reduce batch size:
```bash
--batch-size 32  # or 16 for smaller GPUs
```

### EXLA not finding CUDA
Check environment:
```bash
nvidia-smi  # Should show GPU
echo $CUDA_VISIBLE_DEVICES  # Should be "0" or similar
```

### Slow first epoch
Normal - JIT compilation takes 2-5 min on first batch. Subsequent batches are fast.

### Permission denied on checkpoints
```bash
chmod -R 777 checkpoints/  # On host before mounting
```
