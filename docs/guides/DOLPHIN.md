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

### Local showcase vs a human (nametag + recording)

The 2026-08-07 recipe for filming the production Fox against a human on
the same machine — the LOCAL delay regime, where multishine chains are
strongest (g10b: human local chain 22 vs 4 over netplay). The bot drives
the CSS itself: answers the memory-card dialog, creates/equips the
`EXPH` nametag, picks its character. The human takes port 2 with a real
controller.

```bash
EXPHIL_QUEUE_TRACE=1 XLA_TARGET=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 \
devenv shell -- mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/ms_g10b_human.bin \
  --character fox --frame-delay 3 --deterministic \
  --nametag EXPH \
  --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
  --iso "$HOME/isos/melee.iso" \
  --slippi-port 51442 \
  --replay-dir eval_runs/<session_name>
```

- `--nametag` (max 4 chars) needs memory-card save data for the in-game
  tag list; the script enables a folder-backed card automatically when
  the flag is set. Netplay ignores it (the connect-code tag shows
  instead).
- `--frame-delay 3` is the LOCAL deploy rung (netplay deploys at 4 —
  see the production notes in CLAUDE.md; never deploy untrained ids).
- Record with `wf-recorder -f ~/videos/<name>.mp4` (pick the monitor,
  Ctrl+C to stop); `mkdir -p ~/videos` first or it dies on
  `avio_open failed`.
- Add `--postgame-delay 15` for human sessions: holds the bot's
  autostart for N seconds after each game so you can change character
  at the CSS without racing its START press (added 2026-08-08).

### Direct netplay (bot vs a remote human)

The 2026-08-07 A/B session recipe. Launch AFTER the human side is
already searching (the bot grabs the shared GC adapter otherwise).

```bash
EXPHIL_NETPLAY_HOME=$HOME/.config/slippi-dolphin-bot \
EXPHIL_QUEUE_TRACE=1 XLA_TARGET=cuda12 EXPHIL_GPU_MEMORY_FRACTION=0.25 \
devenv shell -- mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/ms_g10b_human.bin \
  --connect-code 'DBTD#411' \
  --slippi-port 51442 \
  --character fox --frame-delay 4 --deterministic \
  --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
  --iso "$HOME/isos/melee.iso" \
  --replay-dir eval_runs/<session_name>
```

- **Stage control (2026-08-09)**: Direct's game 1 is a RANDOM legal
  stage; the LOSER picks on a real stage screen thereafter. For
  stage-controlled evals pass `--require-stage fd` — wrong-stage games
  are LRAS'd out at frame 1 and the session requeues (~6 draws per
  specific stage; agent inputs are dropped during the quit). In
  multi-game sessions the bot steers the loser-pick screen to its
  configured `--stage` automatically. The 0809 crown decider was
  stage-confounded for want of this flag.
- **Port assignment**: connect order decides whether the bot lands port
  1 or 2 — never assume. `analyze_shine_source` autodetects (netplay
  tag → unique Fox → port 1) since 0809; other analyzers still default
  to port 1.
- Between games: `mix run scripts/analyze_qtrace.exs <log>` — expect the
  sharp lag-agreement peak at `--frame-delay + 2`.
- Teardown: `pkill -f "[l]ibmelee_"` (the Python bridge is gone since
  67533ac; there is no `melee_bridge.py` to kill anymore).
- First game start may briefly stall while shaders warm; the JIT
  sampling graphs are pre-compiled at startup since 2026-08-07 (Agent
  warmup runs the full Policy.sample path).

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
| `--nametag TAG` | nil | In-game Melee nametag (max 4 chars, e.g. `EXPH`) the menu helper creates and equips at the CSS. Local play only — netplay shows the connect-code tag instead. Enables a folder-backed memory card automatically (the tag list needs save data). Added 2026-08-07 for showcase recordings. |
| `--postgame-delay N` | nil | Seconds to hold the bot's autostart after a game ends (counted from leaving the score screen) so a human can change character at the CSS. nil = immediate restart (eval behavior unchanged). |
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
