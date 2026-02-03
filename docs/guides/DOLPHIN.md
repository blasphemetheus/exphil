# Dolphin Integration

Running trained agents against Dolphin/Slippi.

## Prerequisites

### 1. Slippi Dolphin

Download from https://slippi.gg/downloads

AppImage location (after Slippi Launcher install):
```
~/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage
```

### 2. Melee ISO

NTSC 1.02 (Rev 2) required. Configure in Slippi Launcher settings.

### 3. Python Environment

pyenet requires Python <3.13:

```bash
# Install Python 3.12 via asdf
asdf install python 3.12.12
asdf local python 3.12.12

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r priv/python/requirements.txt
```

### 4. System Library: enet

Required for pyenet networking:

```bash
# Arch/Manjaro
sudo pacman -S enet

# Ubuntu/Debian
sudo apt install libenet-dev
```

## Running the Agent

### Async Runner (Recommended for LSTM/temporal)

Separates frame reading from inference. Best for slow models (~500ms LSTM):

```bash
source .venv/bin/activate

mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/lstm_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/path/to/melee.iso \
  --character mewtwo \
  --stage final_destination \
  --on-game-end restart
```

### Sync Runner (For fast MLP models)

Inference on every frame. Best for fast models (<16ms):

```bash
source .venv/bin/activate

mix run scripts/play_dolphin.exs \
  --policy checkpoints/mlp_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/path/to/melee.iso \
  --character mewtwo \
  --action-repeat 3
```

## Command-Line Options

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--policy PATH` | required | Path to exported policy file |
| `--dolphin PATH` | required | Path to Slippi/Dolphin folder |
| `--iso PATH` | required | Path to Melee 1.02 ISO |
| `--port N` | 1 | Agent controller port |
| `--opponent-port N` | 2 | Human/opponent controller port |
| `--character NAME` | mewtwo | Agent character |
| `--stage NAME` | final_destination | Stage |
| `--frame-delay N` | 0 | Simulated online delay |
| `--deterministic` | false | Use argmax instead of sampling |

### Async-only Options

| Option | Default | Description |
|--------|---------|-------------|
| `--on-game-end MODE` | restart | `restart` = auto-start next game, `stop` = exit |

### Sync-only Options

| Option | Default | Description |
|--------|---------|-------------|
| `--action-repeat N` | 1 | Cache action and reuse for N frames |

## Architecture

### Async Runner

Uses Elixir concurrency to decouple slow inference from fast frame reading:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FrameLoop     │────>│   SharedState    │<────│   Inference     │
│   (fast, 60fps) │     │   (ETS table)    │     │   (slow, async) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

- **FrameLoop**: Reads frames at 60fps, sends last known action
- **InferenceLoop**: Runs model async (~2/s for LSTM), updates action when ready
- **ETS table**: Lock-free shared state between processes

This allows smooth 60fps gameplay even with 500ms LSTM inference.

## JIT Warmup

Both scripts include JIT warmup during Step 4:
- Runs dummy inference with zero-filled tensors
- Avoids compilation stutter on first game frame
- Takes ~10s for MLP, ~60s for LSTM on first run

## Wayland Notes (Hyprland, Sway, etc.)

Slippi Dolphin may need environment flags:

```bash
./Slippi-Launcher.AppImage --ozone-platform=wayland
```

## FlashAttention NIF (Experimental)

For attention-based models (sliding_window, jamba), you can enable the FlashAttention NIF for potentially faster inference on Ampere+ GPUs (RTX 30xx/40xx, A100, H100).

```bash
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/attention_policy.bin \
  --dolphin ~/.config/Slippi\ Launcher/netplay \
  --iso ~/path/to/melee.iso \
  --flash-attention-nif
```

**Current Status:**
- ✅ NIF implemented with CPU fallback
- ✅ CUDA kernel written (untested on GPU)
- ⚠️ Integration with Axon models pending (flag parsed but not fully wired)

**When to use:**
- Attention backbone with Ampere+ GPU
- Need lowest possible latency (<1ms attention)

**When NOT to use:**
- MLP/Mamba backbones (no attention)
- CPU-only (NIF has copy overhead, Pure Nx is faster)

See [INFERENCE.md](INFERENCE.md#flashattention-nif) for details.

## Troubleshooting

### libmelee import error

Ensure venv is activated:
```bash
source .venv/bin/activate
```

### pyenet build failure

Install enet system library first:
```bash
sudo pacman -S enet  # Arch/Manjaro
```

### Dolphin not starting

Check paths and permissions:
```bash
ls -la ~/.config/Slippi\ Launcher/netplay/
```

### Frame drops / stuttering

- Use async runner for slow models
- Increase `--action-repeat` for sync runner
- Check CPU usage with `htop`
