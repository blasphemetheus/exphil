# Brief: float-precision input injection so CPU-recorded prefixes replay exactly

> **2026-09-15 review:** the parallel processed-input path is the right
> approach, but the original implementation outline omits necessary fields,
> bridge wiring, and transport lifecycle handling. Read
> [FLOAT_INPUT_INJECTION_REVIEW.md](FLOAT_INPUT_INJECTION_REVIEW.md) before
> using this feature. It supersedes the provisional wire format below.
> Implementation and the full validation/coverage pipeline are complete;
> the review records the results and remaining limitations.

For an Astra instance. Written 2026-09-14 by Claude (Opus 5) from a
reconnaissance of the three repos; nothing below has been implemented.
Bradley's decision: build it. Owner of all three repos: us.

## 1. The problem, precisely

The scenario suite (`scripts/scenario_suite.exs`) reproduces a recorded
moment by starting a fresh game and replaying both ports' recorded
controller inputs frame by frame up to a handoff frame, then handing port
1 to the policy or the scripted teacher. It works only if the replayed game
is bit-identical to the recording up to the handoff.

Melee reads each stick as an 8-bit value and converts it to a float. A
human or bot input therefore always sits on the byte grid. **Melee's CPU
AI does not go through the controller: it writes stick floats directly
into the processed input slot**, so a CPU port's recorded values can be
off-grid. Our replay path can only produce grid values, so a CPU prefix
never reproduces.

Measured (`scripts/compare_replay_inputs.exs`, marth rollout r1 vs its
suite re-recording): port 2 (CPU Marth) main-stick x at frame -39 is
0.0039 in the recording and 0.0062 in the rerun (our 0..1 units; the
nearest grid step). Port 2's position differs by 0.01 at frame -19, the
error compounds, and at frame 112 Marth's attack no longer lands on the
frame it did. Port 1 (the bot, bucketed sticks) matched on every one of
the first 340 frames. Same failure on the older Fox-CPU rollouts. In the
09-14 coverage round 70/77 CPU-sourced handoffs diverged; it is almost
certainly the unexplained "28/36 drifted" of
`docs/planning/RECOVERY_LABEL_CONFIRMATION.md`.

Current workaround (works, keep it as the fallback): `eval_runs/
0914_coverage_round/gen2.sh` regenerates each CPU rollout as an
input-driven game (policy vs a ghost of the recorded CPU inputs, on the
right body); its prefixes replay with zero drift. What it loses: a
REACTIVE opponent in the frames before the handoff.

Slippi's own playback build replays CPU games perfectly because its
playback Gecko codes overwrite the game's PROCESSED inputs from the
recorded floats. That is the mechanism to build into our headless build.

## 2. Where every byte currently lives (verified line numbers)

Three repos, all ours, all siblings of `~/git/exphil`:

**`~/git/libmelee_ex`** (Elixir; the bridge's only input library — the
Python libmelee survives only for menus/dummy).
- `lib/melee/controller.ex:45-100` — `fix_analog_stick/1` snaps a 0..1
  axis to `round((x-0.5)*160)` (the [-80,80] raw grid) and
  `fix_analog_trigger/1` to 140 steps; `tilt_analog/4` (line 152) applies
  them. This is where our own values get gridded, deliberately, so the
  game's processed value is predictable.
- `lib/melee/slippi_pad.ex` — packs a `ControllerState` into Ishiiruka's
  8-byte Slippi pad buffer: byte 0-1 buttons, bytes 2-5 main/C sticks as
  `floor((v-0.5)*254)` signed bytes, bytes 6-7 triggers `trunc(v*255)`.
  `batch/1` frames a `0x01` message: count, then per pad `port + 8 bytes`.
- `lib/melee/console.ex:565` — `state.transport.send(conn, 0,
  SlippiPad.batch(pads), :reliable)` on the direct unix-socket channel
  (`lib/melee/transport/direct.ex`). This is the path the headless build
  uses (`Melee.Bot` default `~/.local/share/slippi/exi-ai-flush/
  dolphin-emu-headless`). The fifo/pipe path (`PRESS A`, `SET MAIN x y`)
  still runs for menus.

**`~/git/slippi-Ishiiruka`** (our Dolphin fork; HEAD 161d7d074; our
commits: `32732106e` direct channel raw events, `7fdaac0d6` direct channel
lockstep pad batches, `161d7d074` pin the EXI device RNG).
- `Source/Core/Core/Slippi/SlippiSpectate.cpp:47-160` —
  `directDrainInputs(block)` reads the socket; message `0x01` = pad batch
  (line 90), copies 8 bytes per port into `m_direct_pad_bufs[port-1]`;
  `directPad(port, out)` (line 147) serves them.
- `Source/Core/Core/HW/EXI_DeviceSlippi.cpp:3259-3310` —
  `prepareOverwriteInputs()`: on each game request it blocks for the
  lockstep batch, then for ports 1..4 writes `1 + 8 pad bytes` (channel pad
  if set, else the pipe device's `SlippiPad`) or `0 + 8 zero bytes` (do not
  overwrite) into `m_read_queue`. Dispatched from `CMD_OVERWRITE_INPUTS =
  0xD9` (`EXI_DeviceSlippi.h:107`, payload length table line 194, handler
  at `.cpp:3509`).
- `Data/Sys/GameSettings/GALE01r2.ini:7378-7440` — `$Optional: Allow Bot
  Input Overrides [Fizzi]`, `C2377598 0000003A #AI/OverwriteInputs/
  OverwriteInputs.asm`: the game-side hook at `0x80377598`. Per frame it
  sends command `0xD9` to the EXI device, reads back the 4 x 9-byte
  answer, and for each port with flag 1 copies the 8 pad bytes into the
  RAW pad status (the code stores 4+4+1 bytes per port with a stride of
  0x0C — verify against the ASM source). Because it writes the RAW
  (pre-conversion) buffer, the byte grid is inherent to this hook.
- Also present and relevant to the alternative in §5:
  `SlippiSavestate.cpp`, `SlippiPlayback.cpp`, `CMD_RECEIVE_INITIAL_RNG =
  0x3A`, `CMD_GET_NEW_SEED = 0xBC`, and our RNG-pin commit.

**Not checked out locally: `project-slippi/slippi-ssbm-asm`** — the
PowerPC sources of every code in the ini, including
`AI/OverwriteInputs/OverwriteInputs.asm` and the Playback codes that
inject processed inputs. Clone it first; do not hand-edit gecko hex.

**`~/git/exphil`** — consumers: `lib/exphil_bridge/melee_port.ex` (bridge
over libmelee_ex), `scripts/scenario_suite.exs` (`prepare_replay/2` at line
92 converts recorded Peppi controllers to bridge inputs via `rec_input/2`;
per-port drift check `drift_check` / `track_drift`; port-2 body from the
replay since 09-14, `--opponent-character` overrides), `ExPhil.Data.Peppi`
(recorded pre-frame floats: `main_stick_x/y`, `c_stick_x/y`, `l_trigger`,
`r_trigger` in 0..1).

## 3. What "done" means

A suite run on a CPU-sourced handoff (e.g. `eval_runs/0914_coverage_round/
rollouts/marth/r1.slp` frame 344, or any entry of
`eval_runs/0914_coverage_round/mined_attempt1` if kept — otherwise re-mine
with `scripts/mine_coverage_handoffs.exs`) replays with **zero drift** and
`scripts/compare_replay_inputs.exs SOURCE RERUN 400` reports no port-2
input mismatch and no state mismatch before the handoff. Then the
09-13/09-14 gates must be unchanged: the delay-4 proof chain
(`eval_runs/0914_delay4_proof/run.sh` steps 4-6 on its existing candidate)
still passes 12/12 cold, 12/12 warm — i.e. the byte path for bot-driven
ports is byte-identical to before when the float path is not requested.

## 4. Design (recommended; three layers, in this order)

**Principle:** add a parallel FLOAT path, never change the byte path. A
port is either byte-driven (today's behaviour, bit-identical) or
float-driven (new). The float path carries the PROCESSED values Slippi
records, so a recorded pre-frame can be replayed as-is.

### Layer A — libmelee_ex (small)
- `Melee.SlippiPad`: a second message type, e.g. `0x02` = float pad batch:
  count, then per pad `port(1) + main_x f32 + main_y f32 + c_x f32 + c_y
  f32 + l f32 + r f32 + buttons(2 bytes as in byte 0-1) + processed-button
  bits if needed` — pick the exact set from what the playback ASM writes
  (see Layer C); big-endian to match the EXI read side.
- `Melee.Controller`: a `set_processed/2` (or `tilt_analog_exact`) that
  stores unsnapped floats and marks the port as float-driven for that
  frame; `Melee.Console` sends `0x02` entries for float-driven ports and
  `0x01` for the rest in the same lockstep step (one frame = one batch of
  each, or one combined message — keep lockstep semantics: the game
  blocks until the batch for this frame arrives, `directDrainInputs(true)`).
- Tests: pack/unpack round trip; `fix_analog_stick` untouched; a
  property that byte-driven ports produce the identical `0x01` bytes as
  before.

### Layer B — Ishiiruka (moderate)
- `SlippiSpectate.cpp`: parse message `0x02` into
  `m_direct_float_pads[port-1]` (+ `_set` flag), cleared per frame like the
  byte pads; `directFloatPad(port, out)`.
- `EXI_DeviceSlippi`: a new command, e.g. `CMD_OVERWRITE_PROCESSED_INPUTS =
  0xDA` (add to the enum and to the payload-length table next to 0xD9),
  whose `prepare…` answers 4 x (flag byte + the float record). Do NOT
  change `prepareOverwriteInputs` (0xD9) semantics: a float-driven port
  should answer flag 0 there (do not overwrite raw) and flag 1 on 0xDA.
- Config: reuse `m_slippiDirectInputs`; no new UI.
- Build: `build-appimage` per the fork's README; install next to
  `~/.local/share/slippi/exi-ai-flush/` as a NEW directory (e.g.
  `exi-ai-float/`) so the existing headless binary the running gates use
  stays untouched until the verdict.

### Layer C — game-side ASM (the hard part)
- Clone `slippi-ssbm-asm`. Read `AI/OverwriteInputs/OverwriteInputs.asm`
  (how 0xD9 is issued and answered) and the Playback codes that write
  processed inputs (search for the per-player input struct writes:
  stick X/Y floats, C-stick, triggers, buttons, and the "physical" fields).
  Establish the exact processed-input addresses and WHEN in the frame
  Melee has finished converting raw pads (the new hook must run after
  that and before the player-controller logic reads them; the playback
  code already solves this ordering — copy its hook location).
- Write `AI/OverwriteInputs/OverwriteProcessedInputs.asm`: send 0xDA, read
  4 x record, for each flag-1 port store the floats into the processed
  struct. Assemble with the repo's build script, append the resulting
  `C2…` gecko to `GALE01r2.ini` under a new `$Optional:` name, enable it in
  the headless config alongside the existing bot-override code.
- Danger: buttons. A processed-input overwrite must be consistent with
  whatever "previous frame" fields Melee derives (press/release edges).
  The playback code handles this; mirror it rather than inventing.

### Layer D — suite integration (small)
- `scenario_suite.exs prepare_replay/2`: emit recorded pre-frame floats
  unsnapped for float-driven ports; a `--float-ports 2` (or automatic: any
  port whose recorded values are off-grid) selects the path per port.
  Record the mode in `agent_runtime` and per run.
- Keep `rec_input/2` for byte ports untouched.

## 5. Alternative route (do not start with it; note for the record)

Run port 2 as an actual CPU during the prefix and seed Melee's RNG from
the recording so the AI re-derives the same decisions. Pieces exist:
`CMD_RECEIVE_INITIAL_RNG`, our RNG-pin commit, Slippi's per-frame recorded
seed. It yields a reactive opponent before the handoff (the float path
does not) but depends on bit-exact state every frame and fails silently
on any drift. Consider only after the float path lands.

## 6. Verification protocol (run in this order, record each)

1. Unit: libmelee_ex tests (`cd ~/git/libmelee_ex && mix test`, 569 tests
   green at a44e57e); new pack/unpack tests.
2. Byte-path regression on the NEW binary: `eval_runs/0914_delay4_proof/
   run.sh`-style cold+warm controls on the existing delay-4 candidate
   (`eval_runs/0914_delay4_proof/round21/candidate.bin`) with the suite
   pointed at the new Dolphin (`--dolphin PATH`): expect 12/12, 12/12,
   zero drift, valid timing. Any difference = the byte path changed; stop.
3. Float path: teacher run (`--driver teacher --audit-teacher-labels
   --response-opponent neutral --window 360 --prefix-history committed
   --reaction-delay 4`) on CPU-sourced handoffs (marth/r1 frames 344,
   1026; the mined_attempt1 list): expect `diverged: false`,
   `first_drift: {}`; `compare_replay_inputs.exs` clean on port 2 through
   the handoff.
4. Then the full coverage pipeline (`collect.sh -> teach.sh -> train.sh`
   with `SOURCES` = the raw rollouts instead of gen2) and compare the
   held-out gates with `docs/planning/COVERAGE_ROUND_2026-09-14.md`
   (24/24 neutral, 22/24 replay).

## 7. Rules that apply

- No `mix` in `~/git/exphil` while a training/eval beam is live
  (`pgrep beam.smp`); libmelee_ex/Ishiiruka builds are fine.
- Long runs via `systemd-run --user … devenv shell -- …` (the Bash tool
  kills anything at ~10 min).
- New Dolphin binary in a NEW install dir; do not overwrite
  `exi-ai-flush`.
- Every experiment directory gets a launcher + `progress.log`; failed
  attempts stay on disk, renamed, never overwritten.
- Commit attribution per the session's system reminder.

## 8. Open facts to establish first (cheap, do before coding)

- Exact fields the playback ASM writes for processed inputs, and the hook
  address/order (from slippi-ssbm-asm).
- Whether Slippi's recorded `l_trigger`/`r_trigger` are processed or
  physical for CPU ports (Peppi exposes both raw and processed for humans;
  confirm which one the CPU writes).
- Whether 0xD9's 4 x 9-byte answer layout can simply be widened (one
  command, both records) or a second command is cleaner — the ini hex
  hard-codes read lengths, so a second command is safer.
- Endianness of floats on the EXI read side (`appendWordToBuffer` writes
  big-endian words; match it).
