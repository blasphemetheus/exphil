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

## slippi-ai head-to-head (ExPhil policy vs vladfi1's agent, one Dolphin)

Added 2026-10-29. Plays an ExPhil policy on port 1 against a slippi-ai
checkpoint (default `~/git/slippi-ai/models/medium-v2`) on port 2 in ONE
local Dolphin, writes `.slp` replays plus a per-game results JSON. Built so
the slippi-ai agent can later be registered as a league opponent.

**Status: the Python worker is verified offline (protocol, model load,
inference timing); the full Dolphin session has NOT been run yet** — it
was written while a GPU training unit held the machine (no `mix`, no
foreground Dolphin). Everything marked *unverified* below is exactly that.

### Design

- **Host = our native harness.** `scripts/play_vs_slippi_ai.exs` is
  `play_dolphin_async.exs` plus an opponent: `MeleePort` (libmelee_ex)
  owns the Dolphin and BOTH pipe controllers (port 2 via
  `dummy_mode: "external"`, the same hook the checkpoint ladder's
  `PolicyOpponent` rides). Every ExPhil deploy knob applies to our side
  unchanged (DEPLOY_KNOBS.md).
- **slippi-ai runs unmodified, in its own venv, as a subprocess.**
  `priv/python/slippi_ai_worker.py` is spawned with
  `~/git/slippi-ai/.venv/bin/python` (TF 2.21 CPU, their libmelee fork
  0.47.3 — nothing shared with our env). Per in-game frame the runner's
  `step_game/3` forwards our `ExPhil.Bridge.GameState` as JSON over an
  Erlang `{:packet, 4}` port; the worker rebuilds a real `melee.GameState`
  and calls their `eval_lib.Agent.step` (their Parser → observation
  filter → async delayed inference → controller decode). A recorder
  object stands in for `melee.Controller`, so their `send_controller`
  writes land in a dict we hand back as the bridge input map.
- Rejected alternative: hosting in THEIR `eval_two.py` with an "exphil"
  player type. It would need a JSON→`GameState` decoder on our side (the
  Python-era one is gone — `MeleePort` is native now), a second transport
  for the beam, and would lose the LatencyProbe/rung machinery and the
  async inference split. The loopback-netplay path (two Dolphins, second
  Slippi login) stays available but was not needed.

### Commands

```bash
cd ~/git/exphil && devenv shell            # worker inherits NIX_PYTHONPATH from this shell

# ExPhil Fox (master ep7) vs medium-v2 Fox, 5 games on FD, windowed
mix run scripts/play_vs_slippi_ai.exs \
  --policy checkpoints/coh_fox_mamba_master_ep7_plain/model_best_policy.bin \
  --character fox --stage final_destination \
  --dolphin "$HOME/.config/Slippi Launcher/netplay-beta-nixos" \
  --iso "$HOME/isos/melee.iso" \
  --num-games 5 \
  --replay-dir eval_runs/1029_vs_slippi_ai/ \
  --results eval_runs/1029_vs_slippi_ai/results.json
```

Add our usual knobs as needed: `--reaction-delay k` (default = the
checkpoint's smallest trained rung), `--deterministic`, `--temperature`,
`--policy-server` (warm agents, see project_jit_warmup), `--headless
--emulation-speed 0 --blocking-input` for the fast sync-equivalent recipe
(the runner paces 60 Hz; *unverified* that the extra port-2 round trip
stays inside the paced frame).

Script-specific flags: `--slippi-ai-model PATH`, `--slippi-ai-python
PATH`, `--slippi-ai-character fox|falco|marth|sheik|jigglypuff|cptfalcon|
peach|yoshi|popo|luigi|pikachu|samus` (medium-v2's set), `--slippi-ai-
temperature T` (their default 1.0), `--slippi-ai-delay N` (see below),
`--num-games N`, `--results PATH`.

Offline smoke of the worker alone (no Dolphin, ~10 s, safe while training
runs): drive it over the packet protocol with synthetic frames — the
2026-10-29 run gave init 2–3 s, mean 3.6 ms per frame round trip (max 5.4
ms) on a machine with a training job active, agent acting on ~56% of
synthetic frames, frame-regression (new game) handled.

### Results JSON

`{meta, summary, games[]}`. Per game: `winner` (`exphil` | `slippi_ai` |
`draw`), `complete` (true when a stock count hit 0; false = left early by
LRAS/timeout/session stop, then ranked by stocks then percent), final
`stocks`/`percent` per side, `last_frame`, `duration_s`. `meta` records
both sides' delays, ports, characters, temperatures and the replay dir.
Replays are the canonical record — rescore from the `.slp` when in doubt.

### Frame delay — who plays at what

- **ExPhil side:** physical reaction delay `k` from `--reaction-delay`
  (async runner: Dolphin frame delay `k-1`, floor `k=1`), measured by the
  LatencyProbe at each game start. master ep7 plain's trained rung is what
  it defaults to.
- **slippi-ai side:** medium-v2 was trained and is released at
  `policy.delay = 21` frames (their deliberate human-reaction handicap).
  The script passes our Dolphin frame delay as their `console_delay`, so
  their internal queue shrinks to `21 - knob` and their TOTAL
  state→action delay stays 21 whatever rung we run — same arithmetic as
  their own `eval_two.py`.
- So a default match is **asymmetric: ours reacts in ~k+1 frames, theirs
  in ~21.** Any win/loss claim carries that asterisk. Three ways to narrow
  it, none verified live:
  1. Slow OUR side: `--local-delay D` holds each of our actions D extra
     frames in the bridge queue (total ≈ k+1+D; e.g. `--reaction-delay 4
     --local-delay 16` ≈ 21). This is an UNTRAINED regime for every
     current checkpoint (delay-id mismatch — pass
     `--allow-latency-mismatch`, expect degraded play). The honest
     symmetric comparison needs a checkpoint trained at that rung.
  2. Speed THEIR side: `--slippi-ai-delay N` overrides
     `state.config.policy.delay` in memory (the analogue of their
     `scripts/update_delay.py`, minus the teacher rename). Also
     off-distribution for their network — they only release d21 for
     medium-v2.
  3. Netplay-rung both: run ours at the netplay recipe (`--reaction-delay
     4` class) and read theirs as "a 21-frame human-handicapped bot".
     Report both numbers, don't average them.

### Known limitations / unverified

- Full session (CSS menuing with the external port-2 controller, replay
  finalization, LRAS, `--num-games` stop at the CSS via `postgame_delay
  5`) is untested end to end; each piece exists in `play_dolphin_async`
  + `PolicyOpponent` runs, the combination does not yet.
- `ExPhil.Bridge.Projectile` has no `spawn_id` (dropped in
  `MeleePort.convert_projectile`), so their `ItemAssigner` gets list
  indices as ids: item SLOT identity across frames is approximate when
  several projectiles coexist (Fox lasers). Fix = plumb `spawn_id` through
  `lib/exphil_bridge/types.ex` + `melee_port.ex` (not done — lib/ frozen
  during the training run).
- Nana (Ice Climbers) is rebuilt from ExPhil's compact Nana (no
  jumps/shield/on_ground) — do not read popo results as faithful.
- Their agent infers on CPU; a slow host could make their delay queue
  drain and `step` block the frame loop (the worker logs `step_ms` every
  30 s and warns above 12 ms). Observed 3.6 ms offline.
- `melee.Stage`/`melee.Character` in their fork are INTERNAL ids; our
  `GameState.stage` is EXTERNAL (converted with `to_internal_stage`),
  our `Player.character` is already internal — both mapped in the worker.
- The worker needs the devenv shell's `NIX_PYTHONPATH`/`PYTHONPATH`
  (numpy etc. live in the devenv profile, not the venv); the script
  passes them through. Launch from `devenv shell`.
- Dropbox "released models" folder (README) is JS-rendered; the listing
  could not be fetched without a browser (2026-10-29). Whether an
  imitation-only (non-RL) checkpoint is published there is UNKNOWN; the
  worker reports `agent_type` (`rl` when `rl_config`/`agent_config` is in
  the pickle, else `imitation`) so any downloaded file self-identifies.
