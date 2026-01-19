# ExPhil Docker Workflow

Quick reference for building, testing, and deploying ExPhil Docker images.

## Images

| Image | Purpose | Base |
|-------|---------|------|
| `exphil:gpu` | Cloud GPU training | `nvidia/cuda:11.8.0` |
| `exphil:cpu` | Local testing | `ubuntu:22.04` |

## Building

```bash
cd ~/git/melee/exphil

# Build GPU image (for cloud deployment)
docker buildx build -f Dockerfile.gpu -t exphil:gpu .

# Build CPU image (for local testing)
docker buildx build -f Dockerfile.cpu -t exphil:cpu .
```

**When to rebuild:**
- Code changes in `lib/`
- Dependency changes in `mix.exs` / `mix.lock`
- Config changes in `config/`
- Script changes in `scripts/`

**Layer caching:** If only `lib/` changed, rebuild is fast (~1-2 min). If `mix.exs` changed, deps recompile (~5-10 min).

## Testing Locally

```bash
# Test help works
docker run --rm exphil:cpu mix run scripts/train_from_replays.exs --help

# Test EXLA loads
docker run --rm exphil:cpu mix run -e "IO.inspect(Nx.default_backend())"

# Test with replays (mount specific subfolder)
docker run --rm \
  -v ~/git/melee/replays/2025-02:/app/replays:ro \
  -v ~/git/melee/exphil/checkpoints:/app/checkpoints \
  exphil:cpu \
  mix run scripts/train_from_replays.exs --epochs 1 --max-files 1 --replays /app/replays
```

**Note:** GPU image (`exphil:gpu`) only works on machines with NVIDIA GPU. Use CPU image for local testing.

## Docker Hub

### Login
```bash
docker login
# Username: bradleyfargo
# Password: <your password or access token>
```

### Push Images
```bash
# Tag for Docker Hub
docker tag exphil:gpu bradleyfargo/exphil:gpu
docker tag exphil:cpu bradleyfargo/exphil:cpu

# Push (overwrites existing tags)
docker push bradleyfargo/exphil:gpu
docker push bradleyfargo/exphil:cpu
```

### Pull Images (on cloud instance)
```bash
docker pull bradleyfargo/exphil:gpu
```

## Cloud Deployment

### Cost Estimates

| GPU | Provider | $/hr | 100 replays, 10 epochs |
|-----|----------|------|------------------------|
| RTX 4090 | RunPod | $0.44 | ~$0.30 (~40 min) |
| A10 | Lambda | $0.60 | ~$0.50 (~50 min) |
| A100 | RunPod | $1.64 | ~$0.55 (~20 min) |
| A100 | Lambda | $1.10 | ~$0.37 (~20 min) |

*Estimates based on ~10x GPU speedup vs CPU for this workload*

### RunPod (Recommended)

#### Account Setup

1. Go to [runpod.io](https://runpod.io) and sign up
2. Go to **Billing** → Add credits ($10-20 is plenty to start)
3. Optional: Add SSH key in **Settings** → **SSH Keys** (makes login easier)

#### GPU Selection

| GPU | $/hr | VRAM | Recommendation |
|-----|------|------|----------------|
| RTX 3090 | $0.22 | 24GB | Budget option, good for testing |
| RTX 4090 | $0.44 | 24GB | **Best value** - fast, affordable |
| A100 PCIe | $1.64 | 40GB | Overkill for your model size |
| H100 | $3.89 | 80GB | Way overkill |

**Recommendation:** Start with **RTX 4090** ($0.44/hr). Your model is small enough that 24GB VRAM is plenty.

#### Pod Configuration

Go to **Pods** → **+ Deploy**

- **Container Image:** `bradleyfargo/exphil:gpu`
- **Container Disk:** 20GB
- **Volume Disk:** 50GB (persists replays/checkpoints between restarts)
- **Volume Mount Path:** `/workspace`
- **Expose HTTP/TCP Ports:** Leave empty (SSH is automatic)
- **Start Command:** Leave blank

**Why use a volume:** Without a volume, replays and checkpoints disappear when the pod stops. Volume persists at ~$0.10/GB/month.

#### Connect to Pod

Once deployed: Pod → **Connect** → **SSH over exposed TCP**

```bash
ssh root@205.234.xxx.xxx -p 12345 -i ~/.ssh/id_ed25519
```

Or use **Web Terminal** button for browser-based access.

#### First-Time Setup on Pod

```bash
# You land in /workspace (persistent volume)
cd /workspace
mkdir -p replays checkpoints

# Verify GPU
nvidia-smi
```

#### Upload Replays

From your **local machine**:

```bash
# Replace IP and PORT with your pod's SSH details
rsync -avz --progress \
  -e "ssh -p PORT" \
  ~/git/melee/replays/ \
  root@IP:/workspace/replays/
```

#### Run Training

On the **pod**:

```bash
cd /app

# Basic training
mix run scripts/train_from_replays.exs \
  --epochs 10 \
  --batch-size 128 \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints/model.axon

# Mamba backbone
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba \
  --epochs 10 \
  --batch-size 64 \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints/mamba_model.axon
```

**Batch size tip:** GPUs handle larger batches (128-256) vs CPU (64). If OOM, reduce batch size.

#### Long-Running Training with tmux

```bash
# Start tmux session (survives disconnects)
tmux new -s training

# Run training inside tmux
mix run scripts/train_from_replays.exs ...

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

#### Download Results

From your **local machine**:

```bash
rsync -avz --progress \
  -e "ssh -p PORT" \
  root@IP:/workspace/checkpoints/ \
  ~/git/melee/exphil/checkpoints/
```

#### Stop the Pod (Important!)

**Don't forget** - you're billed per minute.

- **Stop:** Keeps volume, stops GPU billing. Use if you want to resume later.
- **Terminate:** Deletes everything including volume.

#### Cost Example

Training 235 replays, 10 epochs on RTX 4090:
- Time: ~30-40 min
- Cost: ~$0.30

### Lambda Labs

1. **Create instance** at [lambdalabs.com](https://lambdalabs.com)
   - Choose GPU: A10 ($0.60/hr) or A100 ($1.10/hr)
   - OS: Ubuntu 22.04

2. **SSH and setup Docker with NVIDIA support**:
   ```bash
   ssh ubuntu@<instance-ip>

   # Install Docker
   curl -fsSL https://get.docker.com | sh

   # Install NVIDIA container toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

   # Pull image
   docker pull bradleyfargo/exphil:gpu
   ```

3. **Run with mounted volumes**:
   ```bash
   docker run --gpus all \
     -v ~/replays:/app/replays:ro \
     -v ~/checkpoints:/app/checkpoints:rw \
     bradleyfargo/exphil:gpu \
     mix run scripts/train_from_replays.exs --epochs 10 --replays /app/replays
   ```

### Vast.ai (Budget Option)

1. **Create account** at [vast.ai](https://vast.ai)

2. **Search for instances**:
   - Filter: RTX 3090/4090, Docker available
   - Sort by: $/hr

3. **Rent and connect**:
   ```bash
   ssh -p <port> root@<host>
   ```

4. **Pull and run** (same as Lambda Labs)

## Workflow Diagram

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
│  │ docker build or │───→│ docker run --gpus all           │  │
│  │ docker pull     │    │ bradleyfargo/exphil:gpu         │  │
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

## Training Presets

Presets configure all training options with research-backed defaults. Use presets instead of manually specifying options.

### Preset Selection Guide

| Scenario | Preset | Time on RTX 4090 | Notes |
|----------|--------|------------------|-------|
| Quick GPU validation | `gpu_quick` | ~5 min | 3 epochs, 20 files |
| Standard training | `gpu_standard` | ~30 min | 20 epochs, all augmentation |
| Production quality | `production` | ~2-3 hours | 100 epochs, Mamba, EMA, SGDR |
| Production + online play | `production --online-robust` | ~2-3 hours | + frame delay augmentation |
| Mewtwo specialist | `mewtwo` | ~2-3 hours | 90-frame window |
| Ganondorf specialist | `ganondorf` | ~2-3 hours | 60-frame window |
| Link specialist | `link` | ~2-3 hours | 75-frame window |
| G&W specialist | `gameandwatch` | ~2-3 hours | 45-frame window |
| Zelda specialist | `zelda` | ~2-3 hours | 60-frame window |

### Using Presets

```bash
# Production training (recommended)
mix run scripts/train_from_replays.exs \
  --preset production \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints

# Production for online play (Slippi netplay)
mix run scripts/train_from_replays.exs \
  --preset production \
  --online-robust \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints

# Character-specific training
mix run scripts/train_from_replays.exs \
  --preset mewtwo \
  --character MEWTWO \
  --online-robust \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints
```

### What `--online-robust` Does

Enables frame delay augmentation (0-18 frame random delays per sample) so the model learns to handle Slippi online latency. Without this flag, models work well locally but may struggle with online play.

### Docker Compose Shortcuts

```bash
# Build once
docker-compose build

# Run with presets
docker-compose run train-production        # Production quality
docker-compose run train-production-online # Production + online robust
docker-compose run train-standard          # Balanced speed/quality
docker-compose run train-quick             # Quick GPU test

# Character specialists
docker-compose run train-mewtwo
docker-compose run train-ganondorf
docker-compose run train-link
docker-compose run train-gameandwatch
docker-compose run train-zelda

# Utilities
docker-compose run scan-replays            # Analyze replay collection
docker-compose run find-lr                 # Find optimal learning rate
docker-compose run shell                   # Interactive Elixir shell
```

## Manual Training Commands

For custom configurations beyond presets:

```bash
# Basic training (single-frame MLP)
mix run scripts/train_from_replays.exs \
  --epochs 10 \
  --batch-size 128 \
  --replays /app/replays

# Temporal training (Mamba backbone)
mix run scripts/train_from_replays.exs \
  --temporal \
  --backbone mamba \
  --epochs 10 \
  --batch-size 64

# With Wandb logging
WANDB_API_KEY=your_key mix run scripts/train_from_replays.exs \
  --wandb \
  --wandb-project exphil \
  --epochs 10
```

## Troubleshooting

### Permission denied on docker commands
```bash
# Add yourself to docker group (one-time)
sudo usermod -aG docker $USER
newgrp docker
```

### Hex package timeout during build
Already handled in Dockerfiles with retry and timeout settings:
```dockerfile
ENV HEX_HTTP_CONCURRENCY=1
ENV HEX_HTTP_TIMEOUT=120
RUN mix deps.get --only prod || mix deps.get --only prod
```

### EXLA fails to load (GPU image on CPU machine)
Use CPU image for local testing, or override the backend:
```bash
docker run --rm -e EXLA_TARGET=host exphil:gpu ...
```
Note: This won't fully work because the NIF is compiled for CUDA. Use `exphil:cpu` instead.

### Rustler tries to recompile NIF
The `skip_compilation?: Mix.env() == :prod` setting in `lib/exphil/data/peppi.ex` prevents this. If you see cargo errors, ensure you're running in prod mode (`MIX_ENV=prod`).

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MIX_ENV` | `prod` | Elixir environment |
| `EXLA_TARGET` | `cuda` (GPU) / `host` (CPU) | EXLA backend selection |
| `CUDA_VISIBLE_DEVICES` | `0` | Which GPU to use |
| `WANDB_API_KEY` | - | Weights & Biases API key |

## Files

- `Dockerfile.gpu` - GPU image for cloud training
- `Dockerfile.cpu` - CPU image for local testing
- `docker-compose.yml` - Compose configs (alternative to raw docker commands)
- `.dockerignore` - Excludes build artifacts, keeps image small

## Quick Reference Card

```bash
# === LOCAL MACHINE ===

# Build and push
docker buildx build -f Dockerfile.gpu -t exphil:gpu .
docker tag exphil:gpu bradleyfargo/exphil:gpu
docker push bradleyfargo/exphil:gpu

# Upload replays to RunPod (replace IP/PORT)
rsync -avz --progress -e "ssh -p PORT" \
  ~/git/melee/replays/ root@IP:/workspace/replays/

# Download checkpoints from RunPod
rsync -avz --progress -e "ssh -p PORT" \
  root@IP:/workspace/checkpoints/ ~/git/melee/exphil/checkpoints/


# === RUNPOD ===

# Check GPU
nvidia-smi

# Quick GPU validation (~5 min)
cd /app
mix run scripts/train_from_replays.exs \
  --preset gpu_quick \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints

# Production training (~2-3 hours)
mix run scripts/train_from_replays.exs \
  --preset production \
  --online-robust \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints

# Character specialist (e.g., Mewtwo)
mix run scripts/train_from_replays.exs \
  --preset mewtwo \
  --character MEWTWO \
  --online-robust \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints

# Long runs with tmux
tmux new -s training
# ... run training ...
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```
