# ExPhil Docker Workflow

Quick reference for building, testing, and deploying ExPhil Docker images.

## Images

| Image | Purpose | Base |
|-------|---------|------|
| `exphil:gpu` | Cloud GPU training | `nvidia/cuda:12.6.3-devel` |
| `exphil:cpu` | Local testing | `ubuntu:22.04` |

**Note:** CUDA 12.6+ required for RTX 5090 (Blackwell architecture). RTX 4090 and older also work.

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
| RTX 5090 | ~$0.70 | 32GB | Blackwell arch, requires CUDA 12.6+ |
| A100 PCIe | $1.64 | 40GB | Overkill for your model size |
| H100 | $3.89 | 80GB | Way overkill |

**Recommendation:** Start with **RTX 4090** ($0.44/hr). Your model is small enough that 24GB VRAM is plenty. RTX 5090 works but requires CUDA 12.6+ base image.

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

**Option 1: Cloud Storage (Recommended)**

Best for frequent pod restarts. Upload once, fetch automatically. See [REPLAY_STORAGE.md](REPLAY_STORAGE.md) for full setup.

```bash
# === ONE-TIME SETUP (local machine) ===
# 1. Create Backblaze B2 account and bucket
# 2. Configure rclone:
rclone config  # Add b2 remote with your credentials

# 3. Upload replays:
rclone sync ~/git/melee/slp b2:exphil-replays --progress

# === ON EACH POD ===
# Set env vars in RunPod pod config, or export manually:
export B2_KEY_ID="your_key_id"
export B2_APP_KEY="your_app_key"
export B2_BUCKET="exphil-replays"

# Fetch replays:
/app/scripts/fetch_replays.sh
```

Or set RunPod **Start Command** to auto-fetch on pod start:
```bash
/app/scripts/fetch_replays.sh && sleep infinity
```

**Option 2: Direct TCP (if available)**

From your **local machine**, using SSH over exposed TCP:

```bash
# Replace IP and PORT with your pod's SSH details (from Connect → SSH over exposed TCP)
rsync -avz --progress \
  -e "ssh -p PORT" \
  ~/git/melee/replays/ \
  root@IP:/workspace/replays/
```

**Option 3: Via file sharing service (if rsync/scp fail)**

RunPod's SSH proxy doesn't support rsync/scp subsystems. Use a file sharing service:

```bash
# Local: create archive and upload
cd ~/git/melee/replays
tar czf replays.tar.gz mewtwo/
curl -F "file=@replays.tar.gz" https://0x0.st
# Returns URL like https://0x0.st/Hxyz.tar.gz

# On pod: download and extract
cd /workspace
curl -L https://0x0.st/Hxyz.tar.gz -o replays.tar.gz
tar xzf replays.tar.gz
rm replays.tar.gz
```

**Option 4: Via Jupyter Lab**

If port 8888 is exposed and Jupyter is running, use the web file browser to upload files directly.

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

**ALWAYS use tmux for GPU training.** SSH disconnects are common and will kill your training otherwise.

```bash
# Start a new named tmux session
tmux new -s train

# Inside tmux, run your training
cd /app
mix run scripts/benchmark_architectures.exs --replays /workspace/replays --epochs 3

# Detach from session (training continues in background)
# Press: Ctrl+B, then D

# List running sessions
tmux ls

# Reattach to your session
tmux attach -t train

# Kill a session when done
tmux kill-session -t train
```

**Essential tmux commands:**

| Action | Keys |
|--------|------|
| Detach (leave running) | `Ctrl+B`, then `D` |
| Scroll up | `Ctrl+B`, then `[`, then arrow keys |
| Exit scroll mode | `q` |
| Split horizontal | `Ctrl+B`, then `%` |
| Split vertical | `Ctrl+B`, then `"` |
| Switch panes | `Ctrl+B`, then arrow key |

**Pro tip:** Start tmux FIRST, then run your command. If you forget and your SSH dies, the process dies with it.

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

### RTX 5090 (Blackwell) ptxas errors
```
RuntimeError: No PTX compilation provider is available
ptxas does not support CC 12.0
```
RTX 5090 uses Compute Capability 12.0 (Blackwell architecture). Requires:
1. CUDA 12.6.3+ base image (not 12.2)
2. Use `devel` image (not `runtime`) to include `ptxas` for JIT compilation
3. The current `Dockerfile.gpu` already handles this

### curl/wget not found on pod
The GPU image includes `curl`. If missing, install it:
```bash
apt-get update && apt-get install -y curl
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MIX_ENV` | `prod` | Elixir environment |
| `EXLA_TARGET` | `cuda` (GPU) / `host` (CPU) | EXLA backend selection |
| `CUDA_VISIBLE_DEVICES` | `0` | Which GPU to use |
| `WANDB_API_KEY` | - | Weights & Biases API key |
| `B2_KEY_ID` | - | Backblaze B2 key ID for replay sync |
| `B2_APP_KEY` | - | Backblaze B2 application key |
| `B2_BUCKET` | - | Backblaze B2 bucket name |
| `R2_ACCESS_KEY` | - | Cloudflare R2 access key (alternative to B2) |
| `R2_SECRET_KEY` | - | Cloudflare R2 secret key |
| `R2_ENDPOINT` | - | Cloudflare R2 endpoint URL |
| `R2_BUCKET` | - | Cloudflare R2 bucket name |
| `REPLAY_DIR` | `/workspace/replays` | Target directory for replay sync |

## Files

- `Dockerfile.gpu` - GPU image for cloud training
- `Dockerfile.cpu` - CPU image for local testing
- `docker-compose.yml` - Compose configs (alternative to raw docker commands)
- `.dockerignore` - Excludes build artifacts, keeps image small

## Deployment Checklist

Use this checklist when deploying code changes to GPU for testing/training.

### 1. Build, Tag, Push (Local Machine)

```bash
cd ~/git/melee/exphil

# Build GPU image
docker buildx build -f Dockerfile.gpu -t exphil:gpu .

# Tag for Docker Hub
docker tag exphil:gpu bradleyfargo/exphil:gpu

# Push to registry
docker push bradleyfargo/exphil:gpu
```

### 2. Restart Pod (RunPod Dashboard)

1. Go to [runpod.io/console/pods](https://runpod.io/console/pods)
2. **Stop** the pod (if running)
3. Wait for it to fully stop
4. **Start** the pod (pulls latest image automatically)
5. Connect via SSH or Web Terminal

### 3. Smoke Tests (On Pod)

```bash
cd /app

# Verify compilation (should complete with no errors)
mix compile --warnings-as-errors

# Fetch test dependencies and compile test helpers (first time only)
MIX_ENV=test mix deps.get

# Quick test run (10 tests, ~30 seconds)
MIX_ENV=test mix test --max-cases 10 --exclude slow --exclude integration
```

**Note:** The Docker image is built with `MIX_ENV=prod` which doesn't include test helpers in `test/support/`. You must use `MIX_ENV=test` for all `mix test` commands.

### 4. Full Test Suite (Optional)

```bash
# Run all tests except slow/integration (~1-2 min)
MIX_ENV=test mix test --exclude slow --exclude integration

# Run self-play tests specifically (90 tests)
MIX_ENV=test mix test test/exphil/self_play/ --exclude integration
```

### 5. GPU Architecture Benchmark

```bash
# Quick benchmark (~10-15 min)
./scripts/gpu_benchmark.sh --replays /workspace/replays --max-files 50 --epochs 3

# Thorough benchmark (~30-45 min)
./scripts/gpu_benchmark.sh --replays /workspace/replays --max-files 100 --epochs 5 --batch-size 512
```

**Output:**
- Console table with rankings
- `checkpoints/benchmark_results.json` - Raw data
- `checkpoints/benchmark_report.html` - Visual comparison

### 6. Training (If Everything Passes)

```bash
# RTX 4090 optimized presets
./scripts/gpu_train.sh --preset rtx4090_quick --replays /workspace/replays      # ~5 min test
./scripts/gpu_train.sh --preset rtx4090_standard --replays /workspace/replays   # ~30-60 min
./scripts/gpu_train.sh --preset rtx4090_full --replays /workspace/replays       # ~2-3 hours

# Or generic GPU presets
./scripts/gpu_train.sh --preset gpu_standard --replays /workspace/replays
./scripts/gpu_train.sh --preset production --replays /workspace/replays
```

### Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| Pod doesn't pull new image | Stop fully, then start (not just restart) |
| Compilation errors | Check `git status` locally, ensure all changes committed |
| Test failures | Run locally first with `mix test` before deploying |
| `ExPhil.Test.Helpers` not found | Use `MIX_ENV=test mix deps.get` then `MIX_ENV=test mix test` |
| OOM during benchmark | Reduce `--batch-size` (try 128 or 64), or reduce `--max-files` |
| Pod disconnects during training | Use tmux (see above), check if OOM killed the process |
| Process killed with no error | Likely OOM - check `dmesg \| tail` for "Out of memory" |
| EXLA not using GPU | Verify `EXLA_TARGET=cuda` is set |

## Quick Reference Card

```bash
# === LOCAL MACHINE ===

# Build and push
docker buildx build -f Dockerfile.gpu -t exphil:gpu .
docker tag exphil:gpu bradleyfargo/exphil:gpu
docker push bradleyfargo/exphil:gpu

# Upload replays to cloud storage (one-time)
rclone sync ~/git/melee/slp b2:exphil-replays --progress

# Download checkpoints from cloud storage
rclone sync b2:exphil-replays/checkpoints ~/git/melee/exphil/checkpoints --progress


# === RUNPOD POD CONFIG ===

# Environment Variables (set in RunPod dashboard):
# B2_KEY_ID=your_key_id
# B2_APP_KEY=your_app_key
# B2_BUCKET=exphil-replays

# Start Command (auto-fetch replays on pod start):
# /app/scripts/fetch_replays.sh && sleep infinity


# === RUNPOD (after SSH) ===

# Check GPU
nvidia-smi

# Fetch replays (if not using Start Command)
export B2_KEY_ID="..." B2_APP_KEY="..." B2_BUCKET="exphil-replays"
/app/scripts/fetch_replays.sh

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

# Sync checkpoints back to cloud
rclone sync /workspace/checkpoints b2:$B2_BUCKET/checkpoints --progress

# Long runs with tmux
tmux new -s training
# ... run training ...
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```
