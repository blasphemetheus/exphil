# Dolphin Integration

Running trained agents against Dolphin/Slippi.

## Prerequisites

### 1. Slippi Dolphin — which build depends on windowed vs headless

Download from https://slippi.gg/downloads

**Windowed play** (netplay/stable track, what Slippi Launcher installs by
default):
```
~/.config/Slippi Launcher/netplay/Slippi_Online-x86_64.AppImage
```

**Headless probes need a DIFFERENT build.** `--headless` on the netplay build
fails inside `melee.Console` with `Null video requires mainline or ExiAI
Ishiiruka.` Use the Slippi **mainline (beta track)** Dolphin — adopted
2026-07-18 after it was shown to carry analog trigger presses AND releases
through plain pipes, which the ExiAI fork does not:

```
~/.local/share/slippi/mainline/dolphin-emu-mainline
```

libmelee classifies mainline correctly (Null video allowed, Slippi-section
config, `save_replays`/`replay_dir` supported). Point `--dolphin` at the
wrapper FILE, not a directory — libmelee's path heuristic looks for
"netplay" in a directory name and rejects anything else.

| build | windowed | headless | notes |
|---|---|---|---|
| netplay (stable) | yes | **no** | Launcher default |
| mainline (beta) | yes | yes | required for `--headless` |
| ExiAI Ishiiruka | yes | yes | superseded — drops analog triggers (GOTCHAS #66) |

See GOTCHAS #64 (headless setup), #66-RESOLUTION (why mainline), #69 (pace
the frame loop on unthrottled headless games) and #70 (Launcher beta track
on NixOS needs `APPIMAGE_EXTRACT_AND_RUN=1`).

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

**libmelee must be vladfi1's fork, not upstream PyPI.** `requirements.txt`
pins it, so the command above is enough — but do NOT `pip install melee`,
which silently installs upstream. The bridge needs the fork's stateful
`MenuHelper` and `Console(use_exi_inputs=...)`; upstream gets as far as
launching Dolphin and then dies with `got multiple values for argument
'connect_code'` (and leaks the Dolphin window, since the crash skips
teardown). The bridge preflights this and reports it clearly.

Verify:
```bash
pip show melee | grep -E "Version|Home-page"
# Version: 0.43.0
# Home-page: https://github.com/vladfi1/libmelee
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
| `--stateful-step` | false | O(1) recurrent inference: advance the GRU/LSTM trunk one frame at a time via the Edifice.Stateful step API instead of re-running the full window each frame (temporal `:gru`/`:lstm` policies only). Also enables the agent's `snapshot_state/1` / `restore_state/2` rollback API for netplay. |

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
