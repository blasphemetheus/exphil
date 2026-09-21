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

**Step 10b result (04:50):** `native/exphil_msl` (Rust, rustler +
libloading; loads `libmelee_core.so` from `EXPHIL_SIM_ROOT` at runtime, no
link-time dependency) exposes `open/reset/step/observe/save/restore/
match_config_default` with raw C-struct rows; `ExPhil.Bridge.SimBatch`
(GenServer over `SimBatch.Core`) has the SimPort surface and
`ExPhil.Sim.Env` dispatches by handle (`--backend nif|port`, default nif).
Equivalence: reset + 200 scripted steps bit-identical to the Port path,
frames AND terminals (`sim_batch_test.exs`). Step round trip: **35 µs per
env-frame at any batch (28k env-frames/s)** vs 21k binary Port vs 2.7k JSON;
of the 35 µs, 21 is the sim and ~14 is SimRows decode + GameState
mapping. Search start (64 × 90): port 0.44 s → nif 0.33 s; restore ×64
(58 ms) is now the largest non-sim cost. Ledger:

| 2026-09-21 | 10b NIF | **DONE** | 28k env-frames/s, bit-identical; both backends kept (Port = the comparison Bradley asked for, and the fallback when cargo is absent) |

## Step 8 pilot — BC on random-program oracle labels: REJECTED by the fingerprint bound (2026-09-21 10:35)

Setup: 791 converting oracle episodes (30 warm frames of the prior's own
play + 90 oracle frames, causal pairs) exported by `sim_search.exs
--episodes-out`; `--mix-frames` now runs in BPTT mode as its own cursor
stream (`pipeline.ex`). Two arms, identical otherwise (V3.1 recipe, warm
start from the end of epoch 3, 1,400-file slice, 1 pass, constant 2e-5):
**mix** (+ episodes × 30) and **control**. Drills on one fresh pool (seed
7, 300 starts, 240 frames); fingerprint of the mix arm vs the Dolphin
probe (`eval_runs/0921_step8/`).

| Policy | conv. vs self | conv. vs idle | val (subset split) | fingerprint |
| --- | ---: | ---: | ---: | --- |
| epoch-3 | 0.22 | 0.45 | — | reference |
| control | 0.14 | 0.37 | 1.92 | — |
| mix | 0.27 | 0.40 | 3.14 | **FAILS**: jump_x 0.37→0.08, wavedash/min 4.8→0.4, grabs/min 2.4→8.4, spotdodge→0; NCA distance to Dolphin 7.7 vs within-arm 4.4 |

Reading: a small drill gain vs self, a loss vs idle, and the policy
adopted the search's macro vocabulary (Y-jump, grab, no wavedash). Random
input programs are off the policy's manifold; imitating their best-of-64
teaches the teacher's habits. **Next = policy-guided search**: candidates
are the prior's own samples (temperature ≥ 1), best-of-N by the same
scorer — labels are then things the policy could already do, chosen for
outcome. Keep the control arm in every pilot (the subset pass alone moved
conversion 0.22→0.14).

## Step 8 v1 — BC on POLICY-GUIDED search labels: PASSED (2026-09-21 11:41)

Oracle v1 = the prior's own samples (T = 1.2, 64 per start, 90 frames,
`Search.shoot_policy`), best by the Opening scorer: converting candidate
on **0.88** of the same 1,000 starts (random programs: 0.79), per-candidate
conversion 15 % (vs 10 %), 879 episodes. Same pilot recipe as v0 (warm
start end-of-epoch-3, 1,400-file slice, 1 pass at 2e-5, episodes × 30),
same seed-7 drill pool, same fingerprint gate.

| Policy | conv. vs self | conv. vs idle | val (subset split) | fingerprint gate |
| --- | ---: | ---: | ---: | --- |
| epoch-3 (reference) | 0.22 | 0.45 | — | — |
| control (subset pass only) | 0.14 | 0.37 | 1.92 | — |
| mix v0 (random-program labels) | 0.27 | 0.40 | 3.14 | FAIL |
| **mix2 (policy-guided labels)** | **0.32** | **0.61** | 2.07 | **PASS** (6/6 tells; jump_x 0.37 = 0.37, c-stick aerial 0.59 vs 0.62, short-hop 0.70 vs 0.68; NCA to Dolphin 5.1 vs within-arm 4.4) |

Reading: labels drawn from the policy's own distribution and selected
for outcome raise conversion on both targets (+0.10 vs self, +0.16 vs
idle over the reference; +0.18 / +0.24 over the control that shares
every other knob) at a 0.15-nat replay-likelihood cost and inside the
style bound. Aerials/min 14 → 21 and grabs/min 2.4 → 5.6 are the visible
behavioural shift (more committed offense); lightshield 0.29 → 0.42 is
the one drift to watch. This is expert iteration in one step; the loop
(oracle from mix2's samples → mix3 …) and the full-corpus version are the
V3.2 candidates.

| 2026-09-21 | 8 BC on oracle labels | **DONE — PASSED (v1)** | `checkpoints/fox_v3_1_step8_mix2/model_policy.bin`; `eval_runs/0921_step8/` (pilot.sh, pilot2.sh, drills, fp_mix2/compare.txt) |

## Expert iteration step 2 — mix3: drill compounds, style drifts past the bound (2026-09-21 13:19)

Mix2 played 1,000 fresh starts (seed 11; its own baseline there: opening
0.74, conversion 0.30), its own samples at T = 1.2 were the oracle
(converting candidate on **0.93** of starts, per-candidate 20.7 % — the
sampling distribution itself improved), and mix3 = mix2 + one subset pass
with those 931 episodes × 30 (`eval_runs/0921_step8/{pool_mix2,
oracle_mix2, drill_mix3_*, fp_mix3}`, `checkpoints/fox_v3_1_step8_mix3`).

| Policy (seed-7 pool) | conv. vs self | conv. vs idle | val | fingerprint (8 tells) |
| --- | ---: | ---: | ---: | --- |
| epoch-3 | 0.22 | 0.45 | — | reference |
| mix2 | 0.32 | 0.61 | 2.07 | PASS 8/8-equivalent (6/6 + lightshield 0.42, grabs 5.6 inside) |
| **mix3** | **0.37** | **0.67** | 2.13 | **FAIL 3/8**: short_hop 0.68→0.32, lightshield 0.29→0.51, grabs 2.4→6.2; dashdance 8.4→2.4; NCA to Dolphin 6.2 (mix2 5.1, within-arm 4.4) |

Reading: the loop compounds on the drill metric (+0.05 / +0.06 per
iteration) but the second step traded short hops for full hops and
doubled shielding — full-hop aerials convert more against a slow target
and are not how the corpus plays. The selection pressure is on outcome
only; the style bound is checked after the fact. Levers for iteration 3,
in order: (1) lower the label share (oversample 30 → 10) so the replay
prior anchors harder; (2) sample the oracle at T = 1.0 (T = 1.2 over-
represents rare actions, which is where full hops come from); (3) put the
style term INSIDE the selection — reject candidates outside the tells
(e.g. require a short hop when the human short-hop rate says so) or
tie-break by the prior's own log-likelihood. **Mix2 remains the clean
artifact**; mix3 is the stronger-but-drifted one.

## Iteration 3 — mix3b (from mix2; oracle T = 1.0, episodes × 10): drill holds, drift narrows but persists (2026-09-21 14:53)

Oracle at T = 1.0 on mix2's pool: converting candidate on **0.94** of
starts, per-candidate 24.5 % (T = 1.2: 0.93 / 20.7 %) — lower temperature
lost no coverage. Mix3b val **2.02** (mix2 2.07, mix3 2.13): the lighter
mix moved toward the replay prior.

| Policy (seed-7 pool) | conv. vs self | conv. vs idle | val | fingerprint (8 tells) |
| --- | ---: | ---: | ---: | --- |
| mix2 | 0.32 | 0.61 | 2.07 | pass |
| mix3 (T 1.2, ×30) | 0.37 | 0.67 | 2.13 | fail 3: short-hop 0.32, lightshield 0.51, grabs 6.2 |
| **mix3b (T 1.0, ×10)** | **0.37** | **0.70** | **2.02** | fail 2: lightshield 0.44, grabs 6.2; short-hop 0.36 (bound 0.34 — barely in); NCA 5.15 ≈ mix2 |

Reading: the two knobs fixed the likelihood cost and kept the drill gain,
but the same three habits move in the same direction every iteration
(short hop ↓, shield ↑, grabs ↑). Outcome-only selection against a
non-punishing target systematically prefers full hops, grabs and shield;
this will not wash out with more knob-turning. **Next = the style term
inside the selection** (lever 3): score candidates by outcome AND penalize
the per-episode habit deltas on the eight tells (or hard-reject
candidates whose input stream is off-habit, e.g. full hop where the
short-hop rate says short), so the oracle can only pick human-shaped
winners. Mix2 stays the clean artifact; mix3b is the strongest
"almost-clean" one (2/8 out, both by < 2×).

## Iteration 4 — style term inside the selection: two of three drifts fixed (2026-09-21 16:27)

Oracle (mix2 samples, T = 1.0, style penalty on): converting candidate
0.924 (0.94 without the term); chosen episodes average 0.0 full hops,
0.03 grabs, 0.15 shield frames. Mix4 val **2.01** (best of all arms).

| Policy (seed-7 pool) | conv. vs self | conv. vs idle | val | fingerprint (8 tells) |
| --- | ---: | ---: | ---: | --- |
| mix2 | 0.32 | 0.61 | 2.07 | pass |
| mix3b (T 1.0, ×10) | 0.37 | 0.70 | 2.02 | fail 2 (lightshield 0.44, grabs 6.2) |
| **mix4 (+ style term)** | 0.33 | 0.65 | **2.01** | **fail 1**: lightshield 0.45 (human 0.29); short-hop **0.66** (= human 0.68), grabs **3.9** (in) |

Reading: selection pressure works on what it can see — short hops and
grabs are back in the human range and the drill gain vs idle holds
(0.65 vs 0.61; vs self 0.33 vs 0.32, both inside the ±0.05 noise of 300
starts). Light shield did not move because it is not a selection
artifact: mix2 already sat at 0.42 and every arm inherits it from the warm
history + the subset pass. Next for shielding: a per-frame shield penalty
10x larger, or fix it upstream (the control arm's drift, see the
IMITATION_SQUEEZE pilots). Artifact ranking now: **mix4** = strongest
near-clean (1/8 out), mix2 = clean.

## Regret map v0 (EVALS_PROGRAM item 1) — mix2 on its own 1,000 starts

`scripts/regret_map.exs`, `eval_runs/0921_evals/regret_mix2.json`. Policy
rate = share of its 64 samples that convert; oracle = best of 64. Overall
policy 0.245, oracle 0.94. Regret is ~0.7 in EVERY bin (the oracle
converts almost everywhere), so the informative columns are the policy's
own rate and where the oracle dips: **lowest policy rates** disadvantage
0.105, being_tech_chased 0.104, tech_chase 0.158, advantage 0.17,
def_special 0.175 (and the oracle is weakest there too: 0.80–0.86);
**highest** above_stage 0.35, jc_window 0.35, approach 0.32, def_smash
0.32, def_below 0.31. Reading: the selection gap is largest when the
defender is acting (tech-chase timing, punishing specials) and smallest
in clean approaches. Next: the mix2 − epoch-3 diff on the same pool
(epoch-3 oracle on pool_mix2 queued), and bootstrap intervals.

## Costume head (COACH_STYLE_PRODUCTS S4) — the model picks blue

`scripts/costume_head.exs`, `eval_runs/0921_costume_head/head.bin`:
logistic regression on 62 z-scored fingerprint habits, 42,176 Fox games,
game-level 80/20 split. **Test accuracy 0.468** vs majority 0.348 / chance
0.25 — costume is predictable from play (partly via identity). Weights:
red ← X-jump + Y presses; green ← few Y presses; blue ← c-stick aerials +
R presses; neutral ← c-stick aerials + light shield − bair/dair. The
policies' own sim play: **every checkpoint picks La (blue)** — ep3 p=0.53,
mix2 0.71, mix3b 0.74, mix4 0.71 (9/10 game votes). The expert-iteration
loop made the bot "more blue".

**Correction (17:05, from EVALS_PROGRAM item 2):** with bootstrap
intervals, mix2 − epoch-3 (+0.10 [0.03, 0.17]) and every mix − control are
real; mix3/mix3b/mix4 − mix2 are NOT resolved at 300 starts (+0.01 to
+0.06, CIs cross zero). Iterations 2–4 established the style-term
mechanics, not a compounding gain. Re-run the four arms at 1,000 starts
before claiming a trend.
