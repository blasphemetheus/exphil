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
  **[0, 1] with 0.5 neutral** = `(raw + 80) / 160`, shoulder `raw / 140` — the SAME convention as ExPhil's libmelee-style `ControllerState`, so sticks pass through;
  GOTCHA #123). No conversion; L/R
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
| 2026-09-21 | 4 R1 prior in sim | **DONE — PASSED** | `scripts/sim_prior_play.exs` (V3.1-ep3 vs frozen self, FD, 10 × 1,800 frames, seeds 100-109, 0 agent errors, 56 fps with two GPU agents) + `scripts/sim_r1_compare.exs` vs the Dolphin anon arm (`…_ep3/style_probe`, vs CPU 6): all 6 tells within 2 sd (jump_x_ratio 0.38 vs 0.37, short_hop 0.62 vs 0.68, c-stick aerial 0.47 vs 0.62, aerials/min 10.1 vs 14.2, rolls 1.1 vs 2.8, spotdodge 3.7 vs 3.6); NCA pairwise dolphin-sim 4.83 vs within-arm 4.36 / 4.79; nearest human C2 9/10 (Dolphin 6/10). Caveat: opponent differs (self vs CPU 6) — dashdance/min 15.5 vs 8.4 is the visible effect. `eval_runs/0921_sim_r1/anon_self_n10/compare.txt`. First attempt had every agent walking off stage: GOTCHA #123 (stick convention). |
| 2026-09-21 | 5 fingerprint-from-frames | **DONE** (free) | `StyleFingerprint.fingerprint(states, port, controllers)` already takes frame lists; `sim_prior_play.exs` writes rows in the `style_fingerprint.exs` shape |
| 2026-09-21 | 6 curriculum env v0 | **DONE** | `ExPhil.Sim.Drill` (play-derived pool with history, batched rollouts, `Opening` scorer); baselines: prior vs self 200 starts opening 0.68 / conversion 0.20; 1,000-start run in the overnight chain |
| 2026-09-21 | 7 search-as-teacher v0 | **DONE (tooling)** | `ExPhil.Sim.Search` random shooting, batched; oracle vs idle 10 × 64: converting candidate on 0.80 of starts; 1,000-start + prior-defender runs in the overnight chain; labels written |
| 2026-09-21 | 10a binary rows | **DONE (branch)** | see above; merge after the chain |
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

## Loop profile and optimizations (2026-09-21, before the first long runs)

Measured with `scripts/sim_profile.exs` on V3.1-ep3, then fixed in order:

| Component | Before | After | How |
| --- | --- | --- | --- |
| Policy decisions/s | 113 (batch 1); K agents in parallel processes did NOT scale (118–145 for K=1..8: serialized on the EXLA client) | **8,621** at batch 128 (~14 ms per batched frame, flat in n) | `Agent.batch_init/observe/get_controllers`: one trunk step + one AR sample for n envs. Two hidden O(n) costs found on the way: the per-row embedding (fixed: `embed_states_fast`, one call for the batch) and per-row device reads in the controller conversion (fixed: one device→host copy per head) |
| Savestate restore | 6.3 ms (1 MB base64 over JSON) | 1.25 ms | worker-side cache: `save(keep: true)` / `upload` → `restore({:id, n})` |
| Drill loop (two policies, 32 envs) | 55 fps | **734 fps** | `Drill.rollout_batch`, `Drill.build_pool_from_play_batch` (200-start pool 500 s → 73 s) |
| Search loop (idle defender, 64 candidates as envs) | ~2,800 fps | 1,763–3,000 fps (JSON-bound) | `Search.shoot_batch` |
| Sim step round trip | 337 µs (batch 1), 2.6 ms (batch 8 = 325 µs/env) | → step 10a: binary rows | JSON encode/decode per env is now the bottleneck when a policy is batched |

Step 10a (in progress, worktree `../exphil-boundary`): length-prefixed
frames on the Port (`{:packet, 4}`), the worker sends its numpy dtype
layouts on init, `ExPhil.Bridge.SimRows` decodes gamestate/terminal rows
and encodes controller rows from the descriptor, binary step = `<<1,
controller rows>>` → `<<1, gamestate rows, terminal rows>>`; JSON stays
as the control channel and as the A/B fallback (`binary: false`). This is
the baseline the C-API NIF (step 10b) has to beat.

Scorer change the same night: the drill's primary metric is
`ExPhil.Eval.Opening` (any hit = opening: grab / smash / tilt / aerial /
special, hits detected by percent increase as well as action edges;
converted = second hit before actionable, or grab → throw). FairConversion
(Mewtwo's fair-specific event) stays as a secondary. First batched
baselines: prior vs itself, 200 play-derived starts, 4 s horizon —
opening rate 0.68, conversion 0.20; random-shooting oracle vs idle on 10
starts × 64 candidates — a converting candidate on 0.80 of starts.

**Step 10a result (03:45):** branch `sim-binary-rows` (worktree
`../exphil-boundary`, NOT merged while the overnight chain runs in the
main tree — merge in the morning). `SimPort.step` round trip: batch 1
374 → 70 µs; batch 8 2.9 ms → 0.41 ms; batch 64 23.2 ms → 3.0 ms =
**21,080 env-frames/s (7.6x)**, 47 µs per env-frame against the raw sim's
21 µs. Frames and terminals bit-identical to the JSON path
(`sim_port_test.exs`). The Python side is now ~half of what is left; that
is the NIF's (10b) target: `21 µs` raw step + zero-copy rows.

## Overnight results 2026-09-21 (one shared 1,000-start pool, `eval_runs/0921_sim_drill/self_n1000/pool.{jsonl,term}`)

| Arm | Starts | Opening on ≥1 rollout | Converted | Mean damage | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Prior vs itself, 240-frame horizon | 1,000 | 0.62 | **0.19** | 6.7 | the drill baseline (`…_drill/self_n1000`) |
| Prior vs idle, 240 frames | 1,000 | 0.72 | 0.40 | 9.0 | `…_drill/idle_n1000` |
| Oracle vs idle, 64 candidates × 90 frames | 1,000 | 0.94 | **0.79** | 13.1 | 9.7 % of individual candidates convert; labels for 936 starts (`…_search/idle_n1000/labels.jsonl`) |
| Oracle vs the frozen prior, 64 × 90 | 100 | 1.00 | **0.79** | 12.4 | 5.3 % of candidates; the moving target costs the oracle nothing at the "best of 64" level (`…_search/self_n100`) |

Reading: from the prior's own states, a 90-frame follow-up that opens
and converts exists on ~80 % of starts, against idle or against the prior
itself, while the prior finds one on 19 % (self) / 40 % (idle) with a
240-frame horizon. That gap is the step-8 target (BC/DAgger on the oracle
labels), with the fingerprint bound as the guard. Throughput on the
binary-rows path: search 3,700 fps (2x the JSON run — restore + scoring
now dominate, not the step), prior-defender search 316 s for 100 starts.
