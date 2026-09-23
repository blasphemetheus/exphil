# RL on the imitation prior — self-play for the generalist Fox

**Written 2026-09-20.** Own planning doc with its own goals and status
ledger; independent of the imitation squeeze (`IMITATION_SQUEEZE.md`),
which keeps improving the prior in parallel. Unparks the GOALS.md Track C
prerequisite: "large-scale self-play/PPO" was parked because Dolphin at
60 fps made it hopeless; melee-sim-light (~120k frames/s per core,
bit-exact on our Fox games modulo signed zero, `MELEE_SIM_USES.md`) is
why it unparks.

## What we have and what it is missing

**Have:** a whole-game Fox prior (V3.1-ep3: GRU 512×2 BPTT, AR head,
identity channel) that plays at human option mixes but below human
execution and gets hit more than its teachers (rolls/spotdodges 3–6×).
Imitation cannot fix "gets hit more": the corpus never shows the bot's
own mistakes being corrected. That is exactly what RL adds.

**Have (code, unvalidated):** `ExPhil.Training.PPO` (869 lines),
`self_play/{self_play_env, opponent_pool, league_trainer,
parallel_collector}` (2,300 lines), advantage-weighted BC
(`AdvantageWeighting`, OFFLINE_RL_SPEC.md, shipped 08-12 for multishine).
`PPO_STATUS_2026-07-23.md` is the honest state: the PPO script never ran
end-to-end; the blocker is architectural (actor-critic built as a fresh
MLP instead of on the policy's own temporal trunk; head-map API drift).

**Missing:** (a) a sim ↔ ExPhil adapter (observation embedding and
controller injection against `melee_sim` state instead of libmelee
state); (b) an actor-critic on the GRU trunk with pretrained params
loading by name; (c) a reward with an honest definition; (d) a style /
prior-drift regularizer so RL does not collapse the human-like play the
imitation line bought.

## Goals (the ladder; each gate is pre-registered)

| Gate | Statement | Instrument |
| --- | --- | --- |
| **R0 — sim adapter is faithful** | For a recorded Fox `.slp`, the embedding computed from sim state equals the embedding computed from libmelee state frame-for-frame (modulo signed zero) and controller injection reproduces the recorded inputs | `parity.exs`-style diff on the 288-dim embedding; the 8 bot-vs-CPU games already validated by the sim gate |
| **R1 — prior plays in the sim** | V3.1-ep3 stepped inside the sim against a CPU or itself produces the same fingerprint as in Dolphin (jump ratio, aerial rate, roll rate within the n=10 spread) | `style_fingerprint.exs` on sim-generated `.slp` (the sim writes replays) vs `…_ep3/style_probe/` |
| **R2 — critic learns** | A value head on the frozen trunk, trained on sim rollouts of the prior vs itself, predicts stock/damage outcomes better than the per-state mean (explained variance > 0.3 at 60-frame horizon) | held-out rollouts; this is also COACH_ROADMAP F5 |
| **R3 — RL improves without drift** | PPO with a KL-to-prior penalty beats the frozen prior head-to-head in the sim (win rate > 60 % over ≥ 200 games) while the fingerprint stays within the human range on the identity tells and roll/spotdodge rates *fall* toward the humans | sim league eval + fingerprint bound; the bound is the pre-registered guard against reward hacking |
| **R4 — transfers to Dolphin** | The R3 policy replays R1's live case in Dolphin with the same behaviour stats and no latency/errors | `play_dolphin.exs` live case, `live_report.exs` |
| **R5 — beats a human** | ≥ 1 stock off a human over Direct (= Track D's D5, shared) | Bradley's live look, replays scored |

## Design decisions (made now; revisit only with evidence)

- **Prior stays in the loss.** Objective = PPO clipped surrogate − β·KL(π ∥
  π_prior) with β tuned so the fingerprint bound in R3 holds. Rationale:
  the style products (`COACH_STYLE_PRODUCTS.md`) depend on the policy
  staying human-shaped, and reward hacking in Melee is cheap (camping,
  ledge stalling).
- **Identity channel kept, trained at slot 0 (anonymous) first.** Named
  slots are frozen during RL; a per-player RL fine-tune is a later product
  question.
- **Opponent = frozen prior, then a pool** (`opponent_pool.ex` exists):
  self-play against the latest policy alone forgets; the pool holds the
  prior + checkpoints. Fox ditto first (the only character with a prior).
- **Reward v1 = stock differential + 0.01 × damage differential**, episode
  = one game, no shaping. Event-shaped rewards (fair-conversion, edgeguard)
  come from `ExPhil.Eval.{AerialChain, FairConversion}` only if v1 stalls;
  each shaping term needs its own pre-registered fingerprint check.
- **Reaction/delay rung**: train at reaction 0 like the prior; the delay
  campaign's rung law applies at deploy time unchanged.
- **Python boundary**: the sim is C with a Python API. First cut = a
  Python worker speaking the existing bridge protocol (`priv/python/`), so
  the Dolphin observation code is reused and R0 is a diff, not a rewrite.
  A NIF over the C core is the throughput step after R3 proves the loop.
- **Compute plan**: batch-256 envs per core; the box has 16 cores → ~1.5 M
  fps raw, policy inference on the 5090 is the bottleneck (GRU 512×2 at
  batch 4,096 ≈ ?, measure at R1). Target ≥ 100k policy steps/s before R3.

## Sequencing and ownership

1. **R0 adapter** (ExPhil side, this repo): `ExPhil.Bridge.SimPort` or a
   Python worker; embedding parity test against the 8 validated games.
   Needs from the sim sessions: the Python API surface for stepping with
   raw controller inputs and reading the player/projectile state per frame
   (already used by their validator); the declared arithmetic profile
   (signed zero) so R0 is exact.
2. **Actor-critic on the trunk** (`PPO_STATUS` repair items 1–3): reuse
   `Networks.Policy.build_temporal_trunk`, attach a value head, load the
   prior by name, keep the AR-head sampling path shared with the live
   agent so R4 is free.
3. **R1** in the sim with the prior frozen: fingerprint parity.
4. **R2** critic-only training (also unblocks COACH F5 and the C2 engine
   lens).
5. **R3** PPO + KL, Fox ditto, pool opponents; pre-registered fingerprint
   bound.
6. **R4/R5** back in Dolphin.

Other characters (Mewtwo, G&W) follow once the sim's
`feat/gamewatch-mewtwo` branch is trusted and a Mewtwo prior exists; the
recipe is character-agnostic, the priors are not.

## Risks (named so the ledger can track them)

- Sim non-parity classes (offscreen magnifier tick, historical rounding)
  become training-distribution artifacts that do not transfer (R4 catches
  it; keep an R1 re-check every N updates).
- Reward hacking under KL: camping/ledge-stall still satisfies a loose
  fingerprint bound; add positional-entropy and ledge-time tells to the
  bound before R3.
- Throughput: if policy inference caps at < 20k steps/s, R3 takes days
  per arm; the NIF/batched-inference work moves earlier.
- Forgetting the identity channel: check registry − anon liveness on
  held-out after every RL checkpoint (INVARIANTS 16).

## Status ledger

| Date | Gate | State | Notes |
| --- | --- | --- | --- |
| 2026-09-20 | — | Doc written | Prior = V3.1-ep3; sim gate 8/8 exact on bot-vs-CPU; PPO code unvalidated (PPO_STATUS 07-23); nothing on this ladder run yet |
| 2026-09-20 | R0 | not started | Waiting on: sim Python API surface + arithmetic-profile declaration from the sim sessions |
| 2026-09-23 | R2 | smoke passed | `critic_r2.exs` smoke (8 envs × 300 f × 2 rounds, prior `fox_v3_1_step8_mix4`) runs end to end after two first-run fixes: `Critic.fit` MSE used Kernel `-` on tensors (now `Nx.subtract`), and `critic.bin` is `:erlang.term_to_binary` (Nx.serialize rejects the config strings). Smoke EV is meaningless (4 optimizer steps; 1 stock event). Refit of the smoke data at 60 epochs × batch 128: train EV 0.76, held-out −0.26 (return variance 0.003) — the fit learns; the gate needs the full-size run (`exphil-r2-v1`, 64 envs × 1800 f × 4 rounds). |

## Addendum 2026-09-21 — starting point and the curriculum-env route (Bradley)

Sim `main` now carries every character (Kirby merged as #29; Mewtwo/G&W/
Roy/Pichu via the workspace branches), so the sim is prioritized.

**API facts that set the plan** (`melee_sim/env_batch.py`, `dtypes.py`):
`EnvBatch(batch_size, length)` with `configure_match(stage, players=[
PlayerConfig(character, costume, start_percent, facing, team)])`,
`reset_all/reset_matches`, `step`, `save/restore` per env (bytes),
`gamestate_view` = structured rows with per-slot `pos_x/pos_y`, the five
`speed_*` fields, `percent`, `shield_hp`, `action_id`, `action_frame`,
`hitlag`, `hitstun`, `char_id`, `stocks`, `facing`, `on_ground`,
`jumps_left`, `invulnerable`, plus items and `terminal_view`
(`done/stockout/alive_count`). Controller input via `write_controller`
(float sticks/buttons/shoulder) — the same 13-dim controller ExPhil
emits. Recorded-input replay lives only in the C validator (`native.c`:
raw analog lanes, UCF, physical vs processed buttons); porting that to
Python is a rabbit hole and is NOT needed for the closed loop.

**Starting thing = the closed loop, not input replay.**
1. **State mapper** (`ExPhil.Bridge.SimState`): sim row → `Types.GameState`
   / `Types.Player` (every embedding input has a sim field; `controller_state`
   comes from what we wrote; Nana = second slot on the same port). Unit
   test: the reset-state row of a Fox-vs-Fox FD match maps to the same
   embedding as frame −123 of a Dolphin FD Fox ditto (`parity.exs` style).
2. **Sim worker** over the existing bridge protocol (`priv/python/`): a
   Python process that owns an `EnvBatch`, sends mapped state rows, takes
   controller rows back; batch_size 1 first, then N. This is R0 in
   practice.
3. **R1**: V3.1-ep3 vs a frozen copy of itself in the sim, 30 games,
   fingerprint vs `…_ep3/style_probe/` (the sim writes no `.slp`, so the
   fingerprint runs on a trace → the `StyleFingerprint` input adapter
   takes frames, not files; small change). Since the sim reproduces our
   Dolphin games exactly, R1 failing would point at the mapper, not the
   sim.

**Curriculum envs with combo rewards (Bradley 09-21) — the first RL use.**
`configure_match` + `save/restore` give randomized starts for free:
positions, percents, facing, action + action frame (restore a saved state
then overwrite), stale queue. The reward for "teach this combo" already
exists as scorers: `ExPhil.Eval.AerialChain` (A3) and `FairConversion`
(in-window fair contact consumed) — a drill = (start distribution, scorer,
horizon K). Two teachers on the same drill, in order:
- **Search-as-teacher first** (MELEE_SIM_USES §3): from each start, roll N
  candidate input sequences for K frames with save/restore, keep the one
  the scorer accepts, BC/DAgger on the labels. No critic, no PPO
  machinery, deterministic, and it yields the label machine the drill
  program always wanted. Pre-registered check = the fingerprint bound
  (habits stay human) + the drill's conversion rate on held-out starts.
- **PPO on the same env** once the actor-critic on the trunk exists
  (R2/R3), with the drill scorer as shaped reward and the KL-to-prior
  penalty.
Fox fair-conversion on FD first (scorers exist, prior exists); Mewtwo
short-hop-fair conversion second (needs a Mewtwo prior). 2v1 curricula
(train a doubles-capable agent by facing two opponents; Bradley notes
prior work exists) are possible — `MAX_PLAYERS = 4`, `is_teams`,
`team_alive_mask` — but come after the 1v1 drill loop works.

**Ledger:** R0 → mapper + worker (this repo, Python allowed: bridge
tooling); nothing touches the sim sessions' builds — the installed
package under `~/git/melee-sim-light/.venv` imports cleanly with peppi.
| 2026-09-21 | R0 | **DONE** | SIM_INTEGRATION steps 1-3: mapper embed-identical, worker 1,800 frames 0 errors, row fidelity exact −123..−40 vs Dolphin |
| 2026-09-21 | R1 | **DONE — PASSED** | V3.1-ep3 vs frozen self in the sim, 10 games: all six fingerprint tells within 2 sd of the Dolphin probe arm; NCA dolphin-sim distance = within-arm spread. Loop speed 56 fps with two GPU agents over JSON — the NIF (step 10) is the throughput lever for R2/R3 |
