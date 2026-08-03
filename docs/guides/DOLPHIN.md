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
# NixOS (nixos_slanka):
~/.local/share/slippi/mainline/dolphin-emu-mainline
# Manjaro laptop (Slippi Launcher "Beta" release channel installs it):
~/.config/Slippi Launcher/netplay-beta/Slippi_Netplay_Mainline-x86_64.AppImage
```

On Manjaro, enable the Beta release channel in Slippi Launcher settings —
it installs the mainline AppImage alongside netplay. Do NOT let
AppImageLauncher "integrate" it (that moves it to ~/Applications under a
hashed name, breaking the Launcher's zsync updates and the --dolphin path).

libmelee classifies mainline correctly (Null video allowed, Slippi-section
config, `save_replays`/`replay_dir` supported). Point `--dolphin` at the
wrapper FILE, not a directory — libmelee's path heuristic looks for
"netplay" in a directory name and rejects anything else.

**Mainline headless does NOT write replays to `~/Slippi`** — they land in
the temp User dir, which is deleted at exit. ALWAYS pass `--replay-dir`
on headless runs or the .slp silently vanishes (verified 2026-07-28: a
headless smoke left no new file in ~/Slippi and the eval protocol's
"copy newest" grabbed a stale replay from an earlier windowed run).

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
| `--stateful-step` | false | O(1) recurrent inference: advance the GRU/LSTM trunk one frame at a time via the Edifice.Stateful step API instead of re-running the full window each frame (temporal `:gru`/`:lstm` policies only). Also enables the agent's `snapshot_state/1` / `restore_state/2` rollback API for netplay. Equivalence with the windowed forward is pinned by `stateful_step_equivalence_test.exs` (random frames, toy dims) and `stateful_fixture_parity_test.exs` (fixture-replay windows, deployment dims). |
| `--blocking-input` | false (headless: true) | Dolphin waits for the bot's controller write each frame — the game cannot outrun inference (slippi-ai harness parity). Sync-eval validation: skip stat stays 0 and offset calibration collapses to one peak. |
| `--console-timeout SECS` | 0.1 | Console polling timeout: `console.step` returns no-frame after this instead of blocking forever. This is what lets the LRAS quit sequence complete through the pause screen (the end-of-game `--seconds` SD now pulses L+R+A+Start for a proper Slippi game-end event before the hold-left fallback). `0` = legacy blocking dispatch, which also disables LRAS. |
| `--local-delay N` | 0 | Explicit bridge-side action delay: each policy action applies exactly N frames after the state it answered, via a frame-keyed queue in the bridge (`ActionQueue`, pinned by `priv/python/test_action_queue.py`). Independent of `--frame-delay` (Slippi's native online delay) — the two compose. Enables D≥2 delay experiments and slippi-ai-style console/local budget splitting. SD and dummy inputs are never delayed. |
| `--delay-id-override N` | nil | Force the delay-id one-hot regardless of `--frame-delay`. Deployment rule (2026-08-03): NEVER run a policy at an id it wasn't trained on — override to the nearest trained id (e.g. `ms_g6_sp1` at d4 scores 332 c313 with `--delay-id-override 3` vs 71 at the untrained id 4). Also the interp patch-probe lever. |
| `--stateful-resync N` | nil | With `--stateful-step`: rebuild the hidden state from the buffered window every N frames. Measured 2026-08-03: REJECTED as a deploy default (the periodic state jump breaks chains harder than smooth drift); kept as an experiment knob. |
| `--emulation-speed N` | 1.0 | Emulator throttle. `0` = unthrottled — WITH `--blocking-input` this is the fast headless recipe (menus at max fps, gameplay paced by the frame loop): record-equivalent at half the wall time (2026-08-02). Without blocking input, speed 0 smears gameplay timing — don't. |

### Fast headless eval recipe (2026-08-02)

```bash
--headless --emulation-speed 0 --blocking-input
```
Record-equivalent scores (validated vs windowed and vs speed-1 headless,
3/3 deterministic) at roughly half the block wall time. This is the
default recipe in every 0803+ eval script.

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
