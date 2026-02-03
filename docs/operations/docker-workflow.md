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

#### Persistence Options

Community Cloud pods are **ephemeral** — when you stop them, they often won't be available to resume, and all data is lost when terminated. You have two options for persistence:

**Option 1: Cloud Storage Sync (Recommended for Community Cloud)**

Treat every pod as disposable. State lives in B2/R2, synced automatically.

```
┌─────────────────────────────────────────────────────────────────────┐
│  CLOUD STORAGE (B2/R2) - $5/month for 50GB                          │
│  ├── replays/          ← Upload once from local machine            │
│  └── checkpoints/      ← Auto-sync during training                 │
└─────────────────────────────────────────────────────────────────────┘
           │
           │ fetch on start, sync during training
           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  COMMUNITY POD (ephemeral)                                          │
│  - Spin up any available GPU                                       │
│  - Run setup script → fetches replays                              │
│  - Train with auto-sync → checkpoints backed up every 10 min       │
│  - Pod dies? Spin up another, resume from last checkpoint          │
└─────────────────────────────────────────────────────────────────────┘
```

**Setup:**

1. Set RunPod **Environment Variables** in your pod template:
   ```
   B2_KEY_ID=your_key_id
   B2_APP_KEY=your_app_key
   B2_BUCKET=exphil-replays
   ```

2. Set **Start Command** to auto-fetch:
   ```bash
   /app/scripts/fetch_replays.sh && sleep infinity
   ```

3. After training starts, enable auto-sync in a tmux pane:
   ```bash
   /app/scripts/auto_sync_checkpoints.sh
   ```

**Why this works:**
- Replays download fast (~2-5 min for 10GB on RunPod's network)
- Auto-sync ensures you never lose more than 10 min of training
- Any community pod works — no region lock-in
- Cheaper than Secure Cloud (~50% less)

---

**Option 2: Network Volumes (Secure Cloud Only)**

**Important:** You can *create* Network Volumes in the RunPod dashboard, but Community Cloud pods **cannot attach them**. The "Network Volume" dropdown won't appear when deploying a Community Cloud pod. Only Secure Cloud pods support volume attachment.

**Why Network Volumes (if using Secure Cloud):**
- Persist across pod terminations
- Attach to any Secure Cloud pod in the same region
- Much faster than re-downloading replays each session
- Cost: ~$0.10/GB/month + higher pod prices

**Trade-off:** Secure Cloud pods cost ~2x more than Community Cloud for the same GPU, but you get guaranteed availability and persistent volumes.

**Setup (if using Secure Cloud):**

1. **Create Network Volume:**
   - RunPod Dashboard → **Storage** → **+ New Network Volume**
   - Name: `exphil-workspace`
   - Size: 50GB
   - Region: Pick one and stick with it (e.g., `US-OR-1`)

2. **Deploy Secure Cloud pod in the same region**

3. **Attach volume at `/workspace`**

4. First-time setup:
   ```bash
   /app/scripts/setup_workspace.sh
   ```

---

**Comparison:**

| Aspect | Community Cloud + B2 | Secure Cloud + Network Volume |
|--------|---------------------|-------------------------------|
| Pod availability | Variable (may need to wait) | Guaranteed |
| Data persistence | Cloud storage (sync on start/end) | Network volume (instant) |
| Setup time per session | 2-5 min (fetch replays) | Instant |
| Cost (RTX 4090) | ~$0.44/hr + $5/mo storage | ~$0.69/hr + $5/mo storage |
| Region lock-in | No | Yes |
| Risk of data loss | Low (auto-sync) | Very low |

**Recommendation:** Start with **Community Cloud + B2 sync**. It's cheaper and the auto-sync scripts make it nearly as convenient. Switch to Secure Cloud if you need guaranteed availability for long multi-day training runs.

---

#### Complete Community Cloud Workflow

This is the step-by-step workflow for training on Community Cloud pods with cloud storage persistence.

**One-Time Setup (Local Machine):**

```bash
# 1. Install rclone if you haven't
curl https://rclone.org/install.sh | sudo bash

# 2. Configure Backblaze B2
rclone config
# → n (new remote)
# → name: b2
# → Storage: b2
# → account: YOUR_B2_KEY_ID
# → key: YOUR_B2_APP_KEY
# → (leave rest as defaults)

# 3. Upload your replays to B2
rclone sync ~/git/melee/slp b2:exphil-replays --progress
```

**One-Time Setup (RunPod Dashboard):**

1. Go to **Templates** → **New Template** (or edit your existing one)

2. Set **Container Image:** `bradleyfargo/exphil:gpu`

3. Add **Environment Variables:**
   ```
   B2_KEY_ID=your_key_id_here
   B2_APP_KEY=your_app_key_here
   B2_BUCKET=exphil-replays
   ```

4. Set **Start Command:**
   ```
   /app/scripts/fetch_replays.sh && sleep infinity
   ```

5. Set **Container Disk:** 50GB (for replays + checkpoints)

6. Save the template

**Each Training Session:**

```bash
# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 1: Deploy Pod                                              │
# └─────────────────────────────────────────────────────────────────┘
#
# RunPod Dashboard → Pods → + Deploy
# Select your template, pick an RTX 4090 (or similar)
# Wait for pod to start (~1-2 min)
# Replays auto-fetch during startup via Start Command

# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 2: Connect via SSH                                         │
# └─────────────────────────────────────────────────────────────────┘
#
# Pod → Connect → SSH over exposed TCP
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 3: Start tmux (CRITICAL - protects against disconnects)   │
# └─────────────────────────────────────────────────────────────────┘
tmux new -s train

# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 4: Start auto-sync in a split pane                        │
# └─────────────────────────────────────────────────────────────────┘
# Press Ctrl+B, then % to split vertically
# In the new pane:
/app/scripts/auto_sync_checkpoints.sh

# Press Ctrl+B, then ← to go back to the main pane

# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 5: Verify setup                                            │
# └─────────────────────────────────────────────────────────────────┘
nvidia-smi                              # Check GPU is available
ls /workspace/replays/ | head          # Check replays downloaded

# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 6: Run training                                            │
# └─────────────────────────────────────────────────────────────────┘
cd /app

# Quick test first (~5 min)
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

# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 7: Final sync before stopping pod                          │
# └─────────────────────────────────────────────────────────────────┘
# Auto-sync runs every 10 min, but do a final sync to be safe:
/app/scripts/auto_sync_checkpoints.sh --once

# ┌─────────────────────────────────────────────────────────────────┐
# │ STEP 8: Stop the pod (save money!)                              │
# └─────────────────────────────────────────────────────────────────┘
# RunPod Dashboard → Pods → Stop (or Terminate if done)
```

**Resuming Training (Next Session):**

```bash
# 1. Deploy a new pod (any available GPU in any region)
# 2. SSH in, start tmux
tmux new -s train

# 3. Start auto-sync
tmux split-window -h '/app/scripts/auto_sync_checkpoints.sh'

# 4. Fetch your previous checkpoints from cloud
rclone sync b2:$B2_BUCKET/checkpoints /workspace/checkpoints --progress

# 5. Resume training (it auto-loads the checkpoint)
cd /app
mix run scripts/train_from_replays.exs \
  --preset production \
  --replays /workspace/replays \
  --checkpoint /workspace/checkpoints/model.axon
```

**If SSH Disconnects:**

```bash
# Reconnect to pod
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519

# Reattach to your tmux session (training is still running!)
tmux attach -t train
```

**If Pod Gets Preempted/Terminated:**

Your checkpoints are safe in B2 (auto-synced every 10 min). Just:
1. Deploy a new pod
2. Fetch checkpoints: `rclone sync b2:$B2_BUCKET/checkpoints /workspace/checkpoints`
3. Resume training

---

**Auto-Sync Script:**

Run in the background during training to continuously backup checkpoints:

```bash
# Start auto-sync (syncs every 10 minutes)
/app/scripts/auto_sync_checkpoints.sh &

# Or in a separate tmux pane
tmux split-window -h '/app/scripts/auto_sync_checkpoints.sh'

# Single sync (end of session)
/app/scripts/auto_sync_checkpoints.sh --once
```

See [scripts/auto_sync_checkpoints.sh](../scripts/auto_sync_checkpoints.sh) for details.

---

#### Community Cloud Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPUs available | Try different GPU type (3090 instead of 4090) or different region |
| Replays didn't download | Check env vars are set: `echo $B2_KEY_ID`. Re-run `/app/scripts/fetch_replays.sh` |
| Auto-sync not running | Start it: `/app/scripts/auto_sync_checkpoints.sh &` |
| Lost checkpoint after terminate | Fetch from cloud: `rclone sync b2:$B2_BUCKET/checkpoints /workspace/checkpoints` |
| SSH disconnected | Reconnect and `tmux attach -t train` |
| Training crashed | Check `dmesg | tail` for OOM. Reduce batch size or max-files |
| rclone not configured | Run `/app/scripts/fetch_replays.sh` (auto-configures from env vars) |
| "No cloud storage configured" | Set B2_KEY_ID, B2_APP_KEY, B2_BUCKET in RunPod env vars |

**Common Gotchas:**

1. **Forgot to start tmux** → SSH disconnect kills training. Always `tmux new -s train` first.

2. **Forgot to start auto-sync** → Pod terminates, lose hours of training. Start it immediately after tmux.

3. **Left pod running overnight** → $10+ bill. Set a reminder or use spot instances.

4. **Wrong checkpoint path** → Training starts from scratch. Verify checkpoint exists before training.

5. **Env vars not set** → Replays don't fetch. Check RunPod template has B2_* variables.

#### Pod Configuration

Go to **Pods** → **+ Deploy**

**For Community Cloud (recommended):**
- **Container Image:** `bradleyfargo/exphil:gpu`
- **Container Disk:** 20GB (temporary, lost on terminate)
- **Volume Mount Path:** `/workspace`
- **Expose HTTP/TCP Ports:** Leave empty (SSH is automatic)
- **Start Command:** `/app/scripts/fetch_replays.sh && sleep infinity`

**For Secure Cloud (if using Network Volumes):**
- Same as above, plus:
- **Network Volume:** Select your `exphil-workspace` volume
- **Start Command:** Can leave blank (data persists on volume)

**Note:** The "Network Volume" option only appears for Secure Cloud pods. Community Cloud pods use cloud storage sync instead.

#### Connect to Pod

Once deployed: Pod → **Connect** → **SSH over exposed TCP**

```bash
ssh root@205.234.xxx.xxx -p 12345 -i ~/.ssh/id_ed25519
```

Or use **Web Terminal** button for browser-based access.

#### First-Time Setup on Pod

**Quick setup (recommended):**
```bash
# Run the setup script - creates directories, checks GPU, syncs replays
/app/scripts/setup_workspace.sh
```

**Manual setup:**
```bash
# You land in /workspace (Network Volume)
cd /workspace
mkdir -p replays checkpoints logs

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

#### Updating Code on Pod

When you need to pull code changes without rebuilding the Docker image:

```bash
cd /app

# Install git if not present
apt-get update && apt-get install -y git

# Set up remote (first time only)
git init
git remote add origin https://github.com/blasphemetheus/exphil.git

# Pull latest changes
git fetch origin main
git reset --hard origin/main

# Recompile - DO NOT rm -rf _build!
mix compile --force
```

**⚠️ IMPORTANT: Never `rm -rf _build` on GPU pods!**

The `_build` directory contains compiled EXLA which caches the XLA binary. Deleting it triggers a redownload of the XLA CUDA archive (~500MB-1GB), which can take 10-15+ minutes.

Instead, use `mix compile --force` which recompiles Elixir code without touching the XLA cache.

If XLA does need to redownload, it caches to `~/.cache/xla/`. You can check download progress:
```bash
ls -lah ~/.cache/xla/
```

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
| OOM during benchmark | See [Benchmark Memory Requirements](#benchmark-memory-requirements) below |
| Pod disconnects during training | Use tmux (see above), check if OOM killed the process |
| Process killed with no error | Likely OOM - check `dmesg \| tail` for "Out of memory" |
| EXLA not using GPU | Verify `EXLA_TARGET=cuda` is set |

### Benchmark Memory Requirements

The benchmark script precomputes embeddings to avoid re-embedding each epoch. This trades **memory for speed**, but has limits.

#### Memory Math

For temporal architectures (Mamba, LSTM, etc.):
```
Sequences = frames ÷ stride ≈ frames (stride=1)
Embedding memory = sequences × window_size × embed_dims × 4 bytes

Example (30 replay files, ~175K sequences):
  175,000 × 30 × 1204 × 4 = ~25 GB (embeddings alone)
  + original frame data:     ~8 GB
  + BEAM/Elixir overhead:    ~4 GB
  + XLA buffers:             ~8 GB
  ─────────────────────────────────
  Total:                    ~45 GB minimum
  Safe headroom:            ~64 GB recommended
```

#### Recommended Settings by RAM

| System RAM | Max Files | Sequences | Notes |
|------------|-----------|-----------|-------|
| 32 GB | 8-10 | ~50K | Tight, may swap |
| 64 GB | 20-25 | ~120K | Comfortable |
| 128 GB | 50+ | ~300K | Full dataset |

#### Tradeoff Options

**Option 1: Reduce dataset size (Recommended for benchmarks)**
```bash
mix run scripts/benchmark_architectures.exs --max-files 15
```
- ✅ Simple, immediate
- ✅ Fine for comparing architectures (relative performance is what matters)
- ❌ Less training data

**Option 2: Skip precomputation (embed on-the-fly)**
```bash
# Would require code change to add --no-precompute flag for temporal
```
- ✅ Low peak memory
- ✅ Works on any machine
- ❌ Slower (re-embed each epoch)
- ❌ GPU underutilized during embedding

**Option 3: Use larger RAM pod**
```
RunPod: 128 GB RAM pods available (~$0.50-0.80/hr more)
```
- ✅ No code changes
- ✅ Fastest training
- ❌ More expensive
- ❌ Doesn't scale to huge datasets

**Option 4: Disk-backed embeddings (future)**
```
Write embeddings to NVMe, memory-map during training
```
- ✅ Unlimited dataset size
- ✅ Only limited by disk space
- ❌ Significant code changes
- ❌ I/O can bottleneck training

#### Decision Guide

| Goal | Recommendation |
|------|----------------|
| Quick architecture comparison | `--max-files 15-20` on 64GB pod |
| Thorough benchmark | `--max-files 30-40` on 128GB pod |
| Production training | Don't use benchmark script; use `train_from_replays.exs` which streams data |
| Memory-constrained | Skip temporal archs or use `--skip mamba,jamba,lstm,gru,attention` |

#### Why Benchmark ≠ Training

The benchmark script precomputes ALL embeddings upfront for fair comparison (same data, same preprocessing). Production training (`train_from_replays.exs`) streams data and doesn't have this constraint.

If you just want to train a Mamba model (not benchmark it against others):
```bash
# This streams data, much lower memory requirement
mix run scripts/train_from_replays.exs --temporal --backbone mamba --max-files 50
```

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

# First-time setup (creates dirs, checks GPU, syncs replays)
/app/scripts/setup_workspace.sh

# Or manual setup:
# nvidia-smi                    # Check GPU
# mkdir -p /workspace/{replays,checkpoints,logs}
# export B2_KEY_ID="..." B2_APP_KEY="..." B2_BUCKET="exphil-replays"
# /app/scripts/fetch_replays.sh

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

# Sync checkpoints back to cloud (manual, one-time)
/app/scripts/auto_sync_checkpoints.sh --once

# Or start auto-sync in background (syncs every 10 min)
/app/scripts/auto_sync_checkpoints.sh &

# Long runs with tmux (ALWAYS use tmux for training!)
tmux new -s training
# ... run training ...
# Ctrl+B, D to detach
# tmux attach -t training to reattach

# Pro tip: Run auto-sync in a split pane
tmux split-window -h '/app/scripts/auto_sync_checkpoints.sh'
```

## Training Best Practices

### Logging & Output Management

Training can produce large amounts of output. Manage it effectively:

#### Use `tee` to Log to File While Viewing

```bash
# Log everything to file while still seeing output
mix run scripts/train_from_replays.exs --preset production 2>&1 | tee /workspace/train.log

# Benchmark with logging
mix run scripts/benchmark_architectures.exs --replays /workspace/replays 2>&1 | tee /workspace/benchmark.log

# View just the last 100 lines of a running log
tail -f /workspace/train.log | tail -100
```

#### Use `script` for Full Session Recording

```bash
# Record entire terminal session (including colors, control chars)
script /workspace/session.log

# Now run training...
mix run scripts/train_from_replays.exs --preset production

# Exit script recording
exit

# Replay the session
cat /workspace/session.log
```

#### Redirect Errors Separately

```bash
# Separate stdout and stderr
mix run scripts/train_from_replays.exs \
  --preset production \
  > /workspace/train_stdout.log \
  2> /workspace/train_stderr.log

# Or combine but know which is which
mix run scripts/train_from_replays.exs --preset production 2>&1 | \
  tee >(grep -E "error|Error|ERROR" > /workspace/errors.log) > /workspace/full.log
```

### Increase tmux Scrollback Buffer

Default tmux scrollback is ~2000 lines. Training with many batches can overflow this.

#### Per-Session Increase

```bash
# Inside tmux, increase scrollback for current session
tmux set-option -g history-limit 50000

# Or start tmux with increased limit
tmux set-option -g history-limit 50000 \; new-session -s train
```

#### Permanent Configuration

Create `~/.tmux.conf` on the pod:

```bash
cat > ~/.tmux.conf << 'EOF'
# Increase scrollback buffer
set-option -g history-limit 50000

# Enable mouse scrolling
set -g mouse on

# Better status line
set -g status-right '%H:%M'

# Scroll with mouse wheel
bind -n WheelUpPane if-shell -F -t = "#{mouse_any_flag}" "send-keys -M" "if -Ft= '#{pane_in_mode}' 'send-keys -M' 'copy-mode -e; send-keys -M'"
EOF

# Apply without restarting tmux
tmux source-file ~/.tmux.conf
```

#### Save tmux Buffer to File

```bash
# Inside tmux, save current pane's scrollback to file
# Press Ctrl+B, then :
# Type: capture-pane -pS -10000 > /workspace/tmux_buffer.log

# Or from bash:
tmux capture-pane -pS -10000 -t train > /workspace/tmux_capture.log
```

### Limit Elixir Inspect Output

Error stacktraces can dump huge tensors, overwhelming logs. The benchmark script already includes this fix:

```elixir
# Add at top of script to prevent tensor dumps in errors
Application.put_env(:elixir, :inspect, limit: 10, printable_limit: 100)
```

This truncates inspect output to 10 items and 100 printable chars, preventing multi-megabyte tensor dumps.

#### Runtime Configuration

```bash
# Set via environment variable before running
export ELIXIR_ERL_OPTIONS="+P 1000000"

# Or use IEx for interactive debugging with limits
iex --erl "+P 1000000" -S mix run scripts/train_from_replays.exs
```

### GPU & Memory Monitoring

#### Watch GPU Usage in Real-Time

```bash
# In a second tmux pane (Ctrl+B, then %)
watch -n 1 nvidia-smi

# More detailed with memory breakdown
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1
```

#### Monitor System Memory

```bash
# Real-time memory and CPU
htop

# Just memory, refreshing
watch -n 2 free -h

# Check for OOM kills
dmesg | tail -50 | grep -i "out of memory\|oom\|killed"
```

#### GPU Memory in Training Output

ExPhil shows GPU memory at each epoch. Look for:
```
[12:34:56] GPU: 18.2/24.0 GB (76%)
```

If memory keeps climbing, you may have a leak. Restart training from checkpoint.

### Crash Recovery

#### Check Why Process Died

```bash
# OOM killer logs
dmesg | tail -100 | grep -E "oom|killed|Out of memory"

# If GPU ran out of memory
nvidia-smi  # Shows if GPU is still accessible

# Check if process is still running
pgrep -f "mix run"
```

#### Resume from Checkpoint

ExPhil saves checkpoints periodically. Resume:

```bash
# List available checkpoints
ls -la /workspace/checkpoints/

# Resume from latest (training automatically loads if checkpoint exists)
mix run scripts/train_from_replays.exs \
  --preset production \
  --checkpoint /workspace/checkpoints/model.axon \
  --replays /workspace/replays
```

#### Prevent OOM

```bash
# Reduce batch size (most common fix)
mix run scripts/train_from_replays.exs --preset production --batch-size 64

# Reduce window size for temporal models
mix run scripts/train_from_replays.exs --temporal --window-size 15

# Reduce max files to lower memory footprint
mix run scripts/train_from_replays.exs --preset production --max-files 50

# Use gradient accumulation to simulate larger batches
mix run scripts/train_from_replays.exs \
  --batch-size 32 \
  --gradient-accumulation 4  # Effective batch size: 128
```

### Cost Optimization

#### Use Preemptible/Spot Instances

On RunPod, "Spot" instances are 50-70% cheaper but can be interrupted. Good for:
- Benchmarks
- Quick tests
- Training with frequent checkpoints

Not good for:
- Final production runs (use On-Demand)
- When you need guaranteed completion

#### Checkpoint Frequently

```bash
# Checkpoint every epoch (default for presets)
# Or explicitly set:
mix run scripts/train_from_replays.exs --checkpoint-freq 1

# For long runs, checkpoint every 5 epochs to balance I/O vs safety
mix run scripts/train_from_replays.exs --checkpoint-freq 5
```

#### Stop When Done

**Set a reminder** - GPU costs ~$0.44/hr (RTX 4090). Leaving a pod running overnight = ~$10.

```bash
# In another pane, set a timer
sleep 2h && echo "TRAINING COMPLETE - STOP POD" | wall

# Or auto-stop after training completes
mix run scripts/train_from_replays.exs --preset production && echo "Done!" && \
  curl -X POST "https://api.runpod.io/v1/pods/$POD_ID/stop" -H "Authorization: Bearer $RUNPOD_API_KEY"
```

#### Download Checkpoints to Cloud Storage

Don't lose work if pod terminates:

```bash
# Sync checkpoints to B2 after each major run
rclone sync /workspace/checkpoints b2:$B2_BUCKET/checkpoints --progress

# Or set up automatic sync in background
while true; do
  sleep 600  # Every 10 minutes
  rclone sync /workspace/checkpoints b2:$B2_BUCKET/checkpoints --quiet
done &
```

### Debugging Training Issues

#### JIT Compilation Hangs

First batch triggers XLA JIT compilation. Expected time:
- MLP: 30-60 seconds
- Mamba/LSTM: 2-5 minutes
- Attention: 3-7 minutes

If it takes longer than 10 minutes, something is wrong:
```bash
# Check GPU is being used
nvidia-smi  # Should show process using VRAM

# Check for infinite JIT (dynamic shapes)
# Look for repeated "Compiling..." messages in logs
```

#### Shape Mismatch Errors

Add test batch verification before training (already in benchmark script):
```elixir
# Verify batch shape early
test_batch = Data.batched_sequences(dataset, batch_size: 4, shuffle: false)
|> Enum.take(1)
|> List.first()

IO.puts("Batch states shape: #{inspect(Nx.shape(test_batch.states))}")
# Expected: {4, 30, 1991} for temporal, {4, 1991} for single-frame
```

#### NaN Loss

```bash
# Reduce learning rate
mix run scripts/train_from_replays.exs --learning-rate 1e-5

# Check for data issues
mix run scripts/validate_replays.exs --replays /workspace/replays

# Enable gradient clipping
mix run scripts/train_from_replays.exs --gradient-clip 1.0
```

### Monitoring Dashboard

For long runs, consider setting up a simple monitoring approach:

```bash
# Create a monitoring script
cat > /workspace/monitor.sh << 'EOF'
#!/bin/bash
while true; do
  clear
  echo "=== ExPhil Training Monitor ==="
  echo ""
  echo "GPU Status:"
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
  echo ""
  echo "System Memory:"
  free -h | head -2
  echo ""
  echo "Training Process:"
  pgrep -f "mix run" > /dev/null && echo "Running" || echo "NOT RUNNING"
  echo ""
  echo "Latest Log Lines:"
  tail -5 /workspace/train.log 2>/dev/null || echo "(no log file)"
  echo ""
  echo "Checkpoints:"
  ls -lhtr /workspace/checkpoints/*.axon 2>/dev/null | tail -3 || echo "(none)"
  sleep 5
done
EOF
chmod +x /workspace/monitor.sh

# Run in a tmux pane
./monitor.sh
```

### Pre-Flight Checklist

Before starting a long training run:

```bash
# 1. Verify GPU is available
nvidia-smi

# 2. Check disk space (need ~20GB free)
df -h /workspace

# 3. Verify replays are accessible
ls /workspace/replays/*.slp | head -5

# 4. Test quick run first
mix run scripts/train_from_replays.exs --preset gpu_quick --replays /workspace/replays --dry-run

# 5. Start tmux session
tmux new -s train

# 6. Set up logging
mix run scripts/train_from_replays.exs --preset production --replays /workspace/replays 2>&1 | tee /workspace/train.log

# 7. In another pane (Ctrl+B, %), run monitoring
watch -n 5 nvidia-smi
```
