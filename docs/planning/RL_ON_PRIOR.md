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
| 2026-09-23 | R2 | **PASSED** | Twelve-round rollout dataset, regularized critic selected by validation EV; held-out test EV 0.314. Init: `eval_runs/0923_r2/refit_v2/critic_best.bin`. See `HANDOFF_2026-09-23.md`. |
| 2026-09-23 | R3 | First PPO iteration completed; gate pending | Fixed Polaris `apply_updates` default-nil JIT boundary. Gradient was already working. Added truncation bootstrap and post-update KL guard. Live work: `PPO_LIVE_STATUS_2026-09-23.md`. No win-rate or style improvement established yet. |
| 2026-09-23 | R3 | **Win-rate component passed; style/transfer pending** | 200 PPO iterations completed. Our evaluation: 180W/17L/3D over 200 games vs frozen prior; Claude's separate evaluation: 182W/18L, Wilson 95% CI 86.2–94.2%, all stock-outs. Balanced ports in both. Rolls decreased; spotdodges did not, aerial frequency rose above descriptive human p95. No promotion. See live-status follow-up for current idle state and artifacts. |
| 2026-09-24 | R3 | **Harness control clean** | Prior vs prior through the same evaluator, 200 games: 98W/101L/1D = 49.0 %, Wilson 95 % CI 42.2–55.9 %, port-symmetric (p1 48.0 %, p2 50.0 %), 200/200 by stock-out. The 91.0 % is not an artifact of the evaluator. `eval_runs/0923_ppo/eval_control/summary.json`. |
| 2026-09-24 | R3 | **Style half NOT PASSED** | `scripts/ppo_style_half.py` (read-only, raw habit tells). Identity tells inside the human mean ±2 sd: **3/6**. Scoring the untouched prior through the same harness separates drift from inheritance: **`aerial_per_min` is the only PPO-caused failure** — 36.9/min vs the prior's 20.5 and the humans' 17.0 (+80 % over the prior), the reward-hacking shape the KL bound was pre-registered to catch. `cstick_aerial_frac` (0.084 vs humans 0.581) and `spotdodge_per_min` (3.17 vs 0.85) are **inherited** — the untouched prior is outside the range on both, so they are properties of the prior or the sim harness, not of PPO. R3's directional clause is satisfied on both rates: roll 2.91→1.49 and spotdodge 3.56→3.17, each toward the human mean, and roll actually *entered* the human range the prior was outside. Prescription: rerun at a higher `--kl-coef` (0.05 reached only KL 0.06 at iter 200) and re-score. The learned-metric (NCA) half via `sim_r1_compare.exs` is still unrun. **No promotion.** |
| 2026-09-24 | R3-Mewtwo | **RUNNING** | First non-Fox arm. `--character` added to `ppo_r3`/`critic_r2`/`ppo_eval`/`ppo_r3_eval`; Mewtwo admitted to the sim (`scripts/sim_character_check.exs`, internal id 16 both ports, d=1024, 2000 env-frames/s at 64 envs). Chain `scripts/mewtwo_ppo_chain.sh` under unit `exphil-mewtwo-ppo`: 12-round critic collection → refit → 300 PPO iterations at `--kl-coef 0.01`. Prior = `checkpoints/mewtwo_il_v1_20260924/model_best_policy.bin` (val 3.1043). |
| 2026-09-24 | R2-Mewtwo | **PASSED** | Value head on the frozen Mewtwo trunk. 12 rounds × 64 envs × 1800 f (757 s, ~2000 env-frames/s, d=1024); refit sweep selected stride 6 / hidden 256 / wd 0.001 / dropout 0.0 on a validation round, **test EV 0.61** on a round selection never saw — roughly double Fox's 0.314. Init: `eval_runs/0924_mewtwo_ppo/critic_v1_refit/critic_best.bin`. |
| 2026-09-24 | R3-Mewtwo | **Training COMPLETE; gate pending** | 300 iterations in 6133 s (~20.4 s/iter), 64 envs × 600 f, `--kl-coef 0.01`, critic EV 0.61. Reward/env rose monotonically 0.192 → 0.825 (binned by 25; per-iteration noise hides the trend). KL plateaued ~0.16-0.18, never approached the 0.5 tripwire. **Caveat recorded before the evaluation ran:** from ~iter 200 entropy falls sharply (2.39 peak → 1.45) while KL stays flat — the policy is narrowing inside the imitation manifold rather than leaving it, which against ONE frozen opponent is the signature of an exploit, not of learning Melee. A high sim win rate here is therefore weaker evidence than a moderate one, and the selection phase cannot detect it (same opponent). Arm: `eval_runs/0924_mewtwo_ppo/v1/`, 31 heads. Evaluation (selection/test/control) running under `exphil-mewtwo-eval`. |
| 2026-09-24 | R3-Mewtwo | **GATE SATURATED — frozen-ditto win rate is no longer a usable instrument** | Selection on 60 fresh games each vs the frozen prior: `head_iter70` 57W/3L = 95.0 %, `head_iter150` **60W/0L = 100.0 %**. A 100 % rate has no resolution left: it cannot separate iter 150 from iter 300, nor "better at Melee" from "found an unanswerable exploit against this one opponent". Coherent with the training curve (entropy 2.39 → 1.45 at flat KL ~0.16 = narrowing, not drifting) and with iter 70 already at 95 % while the next 230 iterations raised reward/env 0.30 → 0.825 for almost no additional wins. **An opponent pool (`opponent_pool.ex`, already in the R3 design) is now a prerequisite for any further RL claim, and the same caveat retroactively weakens Fox's 91 %** — same instrument, merely not yet saturated. Remaining useful signals for this arm: SD rate and live Dolphin/human play. |
| 2026-09-24 | R3-Mewtwo | **Win-rate half PASSED with a clean control; mechanism identified** | `head_iter150` (selected on 60 games @seed 8100, tested on 200 FRESH games @seed 9200): **199W/1L = 99.5 %** (CI 97.2–99.9) vs **prior-vs-prior control 100W/98L/2D = 50.0 %** (CI 43.1–56.9). 400/400 by stock-out. Port split p1 99/100, p2 100/100 — R3 trains port 1 only, so full port-2 generality **rules out port-specialization**. **The win rate is the least informative number: deaths/min fell 4.03 → 1.25 (−69 %) while kills/min rose only 4.02 → 4.44 (+10 %).** The entire win is survival — exactly the reported "kills itself a lot, doesn't kill me". Playable export `checkpoints/mewtwo_ppo_v1_iter150_policy.bin` via new `scripts/ppo_export_policy.exs`. **NOT promoted**: the gate is saturated (see previous row), style cannot be scored (no Mewtwo human fingerprint corpus), and transfer is unmeasured. Next evidence must come from Dolphin, a human, and an opponent pool. |
| 2026-09-25 | R3-Mewtwo | **LIVE: iter300 DEGENERATE, iter150 preferred — exploit reading CONFIRMED** | Bradley played both exported heads in Dolphin. iter300 = "rolling to the edge, then grab, then back throw … only really works against a static opponent like what it had to face"; iter150 "definitely better". **The frozen-ditto gate was blind to this** (both 60W/0L in selection; test 99.5 % with a clean 50 % control). **The fingerprints already held the evidence, uncompared:** vs the prior through the same evaluator, iter300 shows roll_backward ×13.1, roll_forward ×4.9, grab ×3.7, throw_back_mix ×4.0, ledge_roll 0 → 0.25, and **aerial_per_min −92 %** (the clincher — a policy throwing no aerials has stopped playing Melee). spotdodge is NOT elevated (0.26 vs 0.39), so the human "spot dodging" impression was wrong and that feature stays out of the signature. iter150 carries the SAME signature at ~half magnitude, so earlier selection mitigates but does not solve, and since the drift is monotone in training it can serve as a live early-stop tripwire alongside `--kl-stop`. **`docs/planning/DEGENERATE_ZOO.md` opened (Entry 1) at Bradley's request.** Key unblock: **degeneracy detection needs NO human corpus** — drift from the PRIOR's own fingerprint suffices, which resolves coordination ASK 12 for this purpose. Open hypothesis (Bradley): Fox behaved better because of more/better/higher-level data — CONFOUNDED by kl-coef 0.01 vs 0.05 and 300 vs 200 iterations; clean test = Mewtwo rerun at 0.05/200 compared on fingerprint drift. |
| 2026-09-25 | R3-Mewtwo-pool | **Pool candidate (`v2/head_iter200`) CLEAN on the zoo Entry 1 signature; playable export exists; human playtest still owed** | Astra's pool arm (`eval_runs/0925_mewtwo_pool/v2`, KL 0.01, 200 iters, opponents = prior + v1 heads 70/150/300 uniform, v1 head100 HELD OUT). Gate: candidate vs prior 99W/1L (saturated, as before); **prior vs held-out head100 5W/95L = 5.0 %; candidate vs head100 66W/34L = 66.0 %** — the first unsaturated Mewtwo number (harness control prior-vs-prior 51/46/3). Degeneracy check run 23:40 with MATCHED baselines (`degeneracy_report_vs_prior.json`, `degeneracy_report_vs_heldout100.json`): `no_known_signature_detected`, zero alerts in both. Drift runs the OPPOSITE way from iter300: vs prior, roll_backward ×0.50, roll_forward ×0.44, grab ×0.41, throw_back_mix 0.06 → 0.00, ledge_roll 0.06 → 0.00; vs head100, grab ×0.26, rolls ×0.60. **aerial_per_min ×0.58 vs prior (just above the ×0.5 review line) and ×0.75 vs head100** — the same reduced-aerial drift v1 iter150 showed, only milder; watch it. Training entropy ended 1.96 (v1 iter300 collapsed to 1.45). Export already written by the pool chain: `checkpoints/mewtwo_ppo_pool_v2_policy.bin` (13:58). **Not promoted, not registered, not playtested** — the checker is a retrospective heuristic and no-signature is not a pass. Next: Bradley plays it (command in HANDOFF_2026-09-25b §7 addendum), then registry. |
