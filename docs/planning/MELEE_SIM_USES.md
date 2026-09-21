# melee-sim-light: how ExPhil can use it

Direction set by Bradley 2026-09-17 (evening): validate the sim on this
machine before/alongside the Fox V3 run; curriculum envs for Mewtwo combos;
RL later. This is the planning doc for that thread. Status: PLANNED — no
ExPhil integration exists yet.

## What it is (local state)

`~/git/melee-sim-light` (kyhavlov; decomp-based, deterministic, batched C
core, `melee_sim` Python/NumPy API). Benchmarks: ~117-121k frames/s per
core at batch 256-512 (9950X3D). All competitive stages. Upstream `main`
excludes Kirby, **Mewtwo**, G&W, Pichu, Roy — but Bradley's local
branches close two of those:

| Branch / worktree | State |
| --- | --- |
| `experiment/blewf-workspace` = `feat/gamewatch-mewtwo` (worktree `~/git/msl-exp`, 4 commits ahead of main, 09-15/16) | **Mewtwo + Mr. Game & Watch admitted with replay validation**; viewer renders articles/hitboxes/shields; G&W silhouettes baked. Docs: `agent_docs/GAMEWATCH_SUPPORT.md`, `MEWTWO_*.md` (move tests, fnmsub captures, teleport captures, erickfm inventory, recording plan). |
| `feat/mewtwo`, `feat/mewtwo-only`, `feat/gamewatch-only` | single-character subsets of the above |
| `feat/arithmetic-profile` (checked out at `~/git/melee-sim-light`) | explicit replay arithmetic profiles — the sim-side twin of ExPhil's `accurate_nmsub` signed-zero profile (`HANDOFF_SIGNED_ZERO_EXPHIL_2026-09-15.md`) |
| `feat/cpu-replay-validation` | desynced CPU Ice Climbers fixture, CPU replay coverage |

Correctness campaign on main (09-08): 503 exact / 26 classified / 0
failures over 5.06 M transitions of recorded replays. Known non-parity:
offscreen magnifier tick timing, historical rounding, missing raw inputs.
The **GameCube adapter** is claimed by the sim's HTML viewer while it is
open — it hogged port 2 during the 09-17 live gate; close the viewer before
any Dolphin session.

## The uses, ranked

The sim's decisive property is not speed; it is **settable state +
determinism**. Dolphin gives observation; the sim gives intervention —
"from this exact state, does the policy convert?" a million times. That
turns correlational evidence about a policy into causal evidence.

### 1. Eval replacement — least compute, least exciting, do FIRST

Replace the Dolphin eval rung wherever no human is involved: policy vs
scripted/CPU-like opponents, thousands of games in minutes, seeded. Today
a 30-second Dolphin game costs 1-2 min wall and a class of harness
failures (orphan Dolphin, adapter contention, CSS wedges, port-1 CPU
toggle). "n >= 8 run-level buckets" stops being a cost. Needs: an
observation adapter (sim state -> ExPhil embedding) and a controller
adapter (policy action -> sim input), both checked against the Dolphin
path on the same recorded prefix. CPU only.

### 2. State-conditioned drills / curriculum envs — highest leverage now

Randomized starts (positions, percents, facing, action + action frame,
stale queue) -> measure conversion. Gives the Mewtwo handoff its
**fair-conversion EVENT** as a distribution rather than a handful of
clips. For BC it is a label machine: a scripted or searched expert converts
from randomized starts -> millions of on-distribution recovery labels —
the recorded-teacher-clip method that closed multishine, minus the
recording bottleneck. Blocked for Mewtwo until `feat/gamewatch-mewtwo`
is trusted (its own validation report exists; re-run it). Fox first.
Embarrassingly parallel CPU.

### 3. Search as the teacher — coolest, medium-high compute

Deterministic + fast => lookahead: from a state, roll N candidate input
sequences for K frames, score (damage, position, stock), keep the best.
An oracle better than any hand-written expert for any supported
character, no hand logic. Distill by BC on oracle labels or DAgger
(oracle corrects the student's own trajectories). Also the sharpest
interp signal we have asked for: the optimal action per state is known, so
policy-vs-oracle divergence is measurable per state class.

### 4. Self-play RL — most compute, the headline, later

PPO at millions of frames/hour (batch-256 envs per core; a 16-core box
~1.5 M fps). The parked GOALS.md track; the sim is why it unparks. It is
downstream of 1 (trusted eval) and 2 (event/reward definitions) — RL has
nothing honest to optimize before those exist.

### 5. Play-time search — least exciting, likely dead end

Lookahead during a live game: 16 ms budget, reaction rung >= 1 frame,
shallow at best, and it is a bespoke decode rule in spirit
(feedback_no_bespoke_decode_rules). Listed to rule out.

| | |
| --- | --- |
| Most exciting / coolest | 3 |
| Highest leverage for the current program | 2 |
| Least exciting | 1 — but it pays for everything else |
| Most compute | 4, then 3 |
| Least compute | 1, then 2 |

## Validation gate before any of it

**Input-replay parity against our own recordings.** Feed a Slippi replay's
processed inputs into the sim from the same initial state; diff player
positions / action states / percents per frame; pass = diverges only where
the README's known non-parity list allows. ExPhil already has the exact
processed-input replay machinery (float-injection suite, `prefix_audit`);
this is that audit pointed at a different engine. Order: Fox on FD/BF
(sync-deterministic stages), then the other four stages, then Mewtwo on
the workspace branch. Then the two adapters (obs, controller) checked
against Dolphin on the same prefix. Only then does use 1 become evidence.

## Open questions

- Which branch is the working base: `feat/arithmetic-profile` (current
  checkout) or the workspace with Mewtwo/G&W? They need to merge for a
  Mewtwo curriculum with the signed-zero profile.
- Python boundary: ExPhil is Elixir; the sim is C with a Python API. Either
  a Port/NIF over the C core or a Python worker speaking the existing
  bridge protocol (`priv/python/`). The bridge protocol route reuses the
  Dolphin harness's observation code.
- Reward/event definitions for 2 and 4 live in ExPhil
  (`MEWTWO_NEUTRAL_TO_COMBO_HANDOFF.md` §10), not in the sim.

## Validation gate — first run 2026-09-19 (`eval_runs/0919_sim_gate/`)

Tool: the sim's own strict validator (`tools/validation/validate_replay.py
--backend native --no-build`, exact projection with classifications, not
tolerances), run from an exphil-side manifest (`manifest.json`, 17 replays
with sha256/provenance; `run_gate.sh`). Other sessions were rebuilding the
sim concurrently, so only existing binaries were used (`--no-build`).

| lane | replays | result |
| --- | ---: | --- |
| **bot vs CPU** (V3 Fox, FD/FoD, CPU 6, Slippi mainline headless, 2026-09-17/18) | 8 | **8/8 PASS, 15,392/15,392 frames exact** with `--diagnostic-signed-zero-equal`; without it 6 FAIL on exactly one field, `speed_y_attack` of the CPU port, `0` vs `-0` (36 rows / 1,924 frames) — the known `fnmsub` signed-zero class (`HUMAN_REPLAY_FAILURES.md`). ~1,450 fps per replay. |
| human erickfm (2020, FD/BF/YS) | 6 | ERROR `replay start scene major is missing` — parser, not parity |
| human Yeti (2020) | 3 | same ERROR |

Reading: for the exact-input lane the sim reproduces our games bit-for-bit
modulo signed zero → uses 1 (eval replacement) and 2 (state-conditioned
drills) are unblocked for Fox on the tested stages. The 2020 corpora fail
before simulation because `native.c` requires `start.scene.major` to
decide whether the Slippi online code set (fnmsub-zero, offscreen damage,
…) governed the match; pre-3.x replays don't carry the scene block. Ask
for the sim sessions: an explicit per-source arithmetic/code-set profile
(`--scene-major` or a manifest field), mirroring exphil's `accurate_nmsub`
declaration — the ambiguity is real (old netplay vs local), so it must be
declared, not inferred. Next: run the human lane once that lands; then
the observation/controller adapters.

## The whole range (Bradley 2026-09-21: "list them all")

Ranked nowhere; labelled by what they are. Every one runs on the same
closed loop (`SIM_INTEGRATION.md` steps 1–2, done).

| Label | Use | What it needs beyond the loop |
| --- | --- | --- |
| **Most professional** | Regression-grade evaluation: every checkpoint scored on fixed start distributions with confidence intervals, in CI, no Dolphin | scorers (have), a run manifest |
| **Highest leverage** | Curriculum drills with combo scorers as reward, search-as-teacher labels | save/restore (have), scorer library |
| **Coolest** | Search-as-teacher: brute-force the best K-frame input sequence from any state, distill it — an oracle for any character with no hand logic | batched rollouts, a scoring horizon |
| **Headline** | Self-play PPO on the imitation prior, millions of frames/hour | actor-critic on the trunk, KL-to-prior |
| **Highest paid** (the product) | The coach: per-situation "what the best players do here" and "what the engine thinks", counterfactual replays of YOUR game (branch from any frame, play out the alternative) | value model (R2), rewind viewer (have) |
| **Most compute** | Full-population league across all 26 characters with matchup tables, on 16 cores × batch 256 | opponent pool (have, unvalidated), scheduler |
| **Least compute** | Deterministic frame-data queries: "does this fair hit from here at this percent?" answered by stepping 30 frames | nothing |
| **Stupidest** (and fun) | Ten-thousand-Fox Monte Carlo of a single situation to make a heat map of where the bot dies; or evolve inputs with a genetic algorithm to find the longest Fox combo on Fox at 0 % | a plot |
| **The Viking one** | Longboat league: every character sails out at once (4 slots, teams), last raft standing; or "berserker" reward = damage dealt only, no survival term, to see what a policy that fears nothing looks like | `is_teams`, 4 slots (have) |
| **Sneakiest** | Reward-hacking zoo: run RL with deliberately bad rewards and catalogue the exploits (ledge stall, camping) so the fingerprint bound is tuned before the real run | fingerprint-from-frames (step 5) |
| **Most scientific** | Style transfer test: same drill, every registry profile, measure which habits survive RL pressure — is identity a prior or a costume? | identity channel (have) |
| **Most useful to humans** | "Play this moment 100 times": load a real replay frame, let a human retry it against the bot from that state (Improoover reps without the savestate plumbing) | replay-to-state seeding (step 3) |
| **Most Melee** | A 20XX-style practice partner: CPU that DI's like SKWA or C2, tech-chases like a human, shield-drops like a human — the named profiles as training dummies | named profiles (have) + drill starts |
| **Longest shot** | Cross-character transfer via the sim: train Fox, fine-tune Mewtwo on sim-generated Mewtwo games labeled by the search oracle | Mewtwo admission trusted |
