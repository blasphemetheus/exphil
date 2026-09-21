# melee-sim-light integration — the overarching plan and its tracker

**Written 2026-09-21.** One document for everything ExPhil does with the
sim: the plan, the boundary decisions, and a checklist of how done each
piece is. Consumers: `RL_ON_PRIOR.md` (self-play, curriculum envs),
`COACH_STYLE_PRODUCTS.md` (curriculum reps, engine lens),
`IMITATION_SQUEEZE.md` (search-as-teacher labels), `MELEE_SIM_USES.md`
(the ranked uses, validation gate). Update the ledger at the bottom
whenever a box moves.

## Decision: wrap, never port (Bradley, 2026-09-21)

The sim is a decompiled Melee engine (C core + extracted game data +
replay-locked validation, 5 M transitions). Its value is the provenance
chain to the real game. ExPhil consumes it as a library and never
re-implements game rules; an Elixir port would be a second engine to
maintain forever and would only ever be measured against the first.

What ExPhil owns (Elixir): the state mapper, the embedding, drill
definitions, scorers/rewards, the search-as-teacher loop, training. What
the sim owns: physics, characters, stages, items, determinism,
save/restore. The boundary is the `EnvBatch` API (Python) now and the C
batch API (NIF) when throughput demands it.

## The plan (ordered; each step has a gate)

| # | Step | Gate | Owner |
| --- | --- | --- | --- |
| 1 | **State mapper** `ExPhil.Bridge.SimState`: sim gamestate row → `GameState`/`Player`; `ControllerState` → sim controller row | unit tests pin id spaces + a mapped state embeds bit-identically to the same state built the Peppi way | exphil |
| 2 | **Sim worker** (`priv/python/sim_worker.py`): owns one `EnvBatch`, speaks the existing bridge protocol (state rows out, controller rows in); batch 1, then N | `ExPhil.Bridge.SimPort` steps a Fox ditto on FD for 1,800 frames with 0 protocol errors; per-frame latency measured | exphil |
| 3 | **Row-fidelity check against Dolphin**: a real Dolphin game's first frames vs the sim's reset + first frames for the same config (positions, action ids, stocks) | field-for-field equality on frame −123..−100 (pre-input frames are deterministic) | exphil |
| 4 | **R1 — prior plays in the sim**: V3.1-ep3 vs a frozen copy, 30 games; fingerprint vs Dolphin probe games | identity tells and roll/aerial rates within the n=10 spread of `…_ep3/style_probe/` | exphil |
| 5 | **Fingerprint-from-frames adapter**: `StyleFingerprint` accepts frame lists (the sim writes no `.slp`) | same numbers as the `.slp` path on a recorded game | exphil |
| 6 | **Curriculum env v0**: randomized starts via `configure_match` + `save/restore`; drill = (start distribution, scorer, horizon); Fox fair-conversion on FD with `FairConversion`/`AerialChain` as scorer | conversion rate of the frozen prior measured on 1,000 starts (the baseline every teacher must beat) | exphil |
| 7 | **Search-as-teacher v0**: N candidate input sequences × K frames from each start, keep scorer-accepted; emit labels | oracle conversion rate ≫ prior's; labels replay in the sim | exphil |
| 8 | **BC/DAgger on oracle labels** → policy; fingerprint bound + held-out conversion | drill conversion ↑ with fingerprint inside the human range | exphil (`IMITATION_SQUEEZE` pilot shape) |
| 9 | **Actor-critic on the trunk** (PPO_STATUS repair) → R2 critic, R3 PPO+KL on the same env | `RL_ON_PRIOR.md` R2/R3 | exphil |
| 10 | **NIF over the C batch API** when policy-step throughput, not sim speed, is the bottleneck | ≥ 100k policy steps/s | exphil |
| 11 | **Human-lane validation** of 2020 corpora (needs the declared scene/code-set profile) | `eval_runs/0919_sim_gate/` human lane passes | sim sessions |
| 12 | Mewtwo / G&W drills once their sim admission is trusted and a prior exists | per-character R1 | both |

## Boundary facts (pinned; re-verify if the sim's dtypes change)

- Rows: `melee_sim/dtypes.py` — `gamestate_dtype` = `frame_id`,
  `stage_id`, `num_players`, `stage{randall, fod_platforms{left,right}}`,
  `slots[4]` (`present`, `source_player`, `team_relation`, `pos_*`, five
  `speed_*`, `percent`, `shield_hp`, `action_id`, `action_frame`, `hitlag`,
  `hitstun`, `char_id`, `stocks`, `facing`, `on_ground`, `jumps_left`,
  `hurtbox_state`, `invulnerable`), `items[15]`; `terminal_view` =
  `done`, `match_ended`, `stockout`, `alive_count`.
- Id spaces: `char_id` = internal fighter kind = ExPhil's frame-level
  character id (identity, clamp > 0x20 → 32; NIF `internal_character_id`);
  `stage_id` = external Slippi id = `GameState.stage` (GOTCHA #96);
  `action_id` = GALE01 action state = `Player.action`.
- Controller in: `write_controller` float row (buttons 0/1, sticks in
  **[0, 1] with 0.5 neutral** = `(raw + 80) / 160`, shoulder `raw / 140`;
  GOTCHA #123). `SimState.axis/1` converts from libmelee [-1, 1]; L/R
  collapse to `max`.
- Recorded-input replay exists only in the C validator (`native.c`: raw
  analog lanes, UCF, physical vs processed buttons). Not needed for the
  closed loop; do not port.
- Determinism: bit-exact vs our Dolphin games modulo signed zero
  (`--diagnostic-signed-zero-equal`), 8/8 bot-vs-CPU games.
- Package: `~/git/melee-sim-light/.venv` imports `melee_sim` + `peppi_py`;
  the checkout is the sim sessions' workspace — read-only for us, never
  rebuild it.

## Open questions (answer at the step that needs them)

- Nana: is the follower a second slot with the same `source_player`? (the mapper assumes yes; verify with an ICs match at step 4 — step 3 used Fox)
  mapper assumes yes; verify at step 3 with an ICs match)
- `facing` u1: verified at step 2 (P1 spawns at x = −60 with facing 1, toward center) — 1 = right.
- Items vs projectiles: Fox lasers are items in the sim; ExPhil's
  `projectiles` list is empty from the mapper. Does the embedding read
  projectiles? If so, map laser item types at step 4.
- Stadium transformation / Whispy are not in the row; FD/BF first.
- ~~Which sim branch is the base~~ — DECIDED: `main` (see recipe below).
  or the workspace? Decide with the sim sessions before step 2 pins the
  venv.

## Ledger

| Date | Step | State | Evidence |
| --- | --- | --- | --- |
| 2026-09-21 | 1 mapper | **DONE** | `lib/exphil_bridge/sim_state.ex`; `test/exphil_bridge/sim_state_test.exs` 7/7 (field-for-field vs Peppi convention, atom/string keys, loud KeyError, Nana fold, id clamp, controller row, embed-identical) |
| 2026-09-21 | 2 worker | **DONE** | `priv/python/sim_worker.py` + `ExPhil.Bridge.SimPort`; `scripts/sim_smoke.exs`: 1,800 frames, 0 protocol errors, round trip mean 354 µs (p99 471), dash-dance script produces DASHING/TURN, save/restore round trip 1.04 MB; `sim_port_test.exs` 3/3 (`--include external`). Found GOTCHA #123 (stick axes [0,1]). |
| 2026-09-21 | 3 row fidelity | **DONE** | `scripts/sim_row_fidelity.exs` on `…_ep3/style_probe/anon/g1` (Fox c1 vs Fox c0, FD, seed from the .slp): 0 mismatches on 16 fields × 2 ports over −123..−40 (84 frames); first divergence −39 = the CPU's first input (walk). Found GOTCHA #124 (reset row labeled one frame early); mapper subtracts 1. |
| 2026-09-21 | 11 human lane | blocked | waiting on the sim's declared scene profile |

## Base and build recipe (decided 2026-09-21: `main` is the base)

Our clone: `~/git/msl-main` (origin = kyhavlov/melee-sim-light, branch
`main`, currently 5e036b4a with Kirby merged). Never touch the sim
sessions' checkout at `~/git/melee-sim-light`; we only borrow its venv
interpreter and its already-built PPC toolchain (read-only symlink).

    git clone --branch main ~/git/melee-sim-light ~/git/msl-main   # then set origin to GitHub
    ln -s ~/isos/melee.iso ~/git/msl-main/SSBM.iso
    PYV=~/git/melee-sim-light/.venv/bin/python
    $PYV -m tools.data.extract --iso ~/isos/melee.iso --out-dir ~/git/msl-main/data   # ~20 s
    ln -s ~/git/melee-sim-light/build/melee_core/toolchain ~/git/msl-main/build/melee_core/toolchain
    make python-library PY=$PYV HOST_CC=gcc -j16    # from the exphil devenv PATH (gcc 15); ~40 s

`SimPort` reads `EXPHIL_SIM_ROOT` (default `~/git/msl-main`) and
`EXPHIL_SIM_PYTHON` (default the sim venv python) and exports
`PYTHONPATH`, `MSL_CORE_LIBRARY`, `MSL_DATA_DIR` for the worker. The
extracted data profile is per-branch: the workspace's `build/data-main`
is NOT readable by main's tools (manifest profile mismatch) — extract per
clone. `uv` is not on this box; the Makefile's `PY=` override sidesteps it.

Verified on main at step 2: FD Fox ditto resets at frame −123, P1 at
(−60, 10) facing 1 (= right, toward center), P2 at (60, 10) facing 0;
`MatchConfig` seed field is `seed`; `PlayerConfig` has `team_id`, not
`team`; raw single-env step ≈ 21 µs, JSON round trip ≈ 350 µs (the NIF is
step 10 for a reason).
