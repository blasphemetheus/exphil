# Sim parity branch — scope (2026-10-06)

**Goal.** Seed the sim bit-exactly from the *training corpus* (erickfm ranked
netplay, Slippi 2.0.1, 2020) so real DAgger — "what would the human have
done in the bot's own state" — is available whenever we want it. Today
`ExPhil.Sim.Seed` is bit-exact on our 2026 Dolphin recordings (Slippi
3.15–3.19) and diverges within a few hundred frames on 38/40 ranked games
(INPUT_COHERENCE "10-05 15:40"). Bradley (10-06): make our own branch of
the sim for this; upstream it if it is useful to them, keep it if it costs
efficiency.

**Has this been done on a branch already? No.** Checked `~/git/melee-sim-light`
(all `experiment/blewf-workspace*`, `feat/*`, `fix/*`, `legacy-sim`) and
`~/git/msl-main` (`exphil-replay-step` = only our seeding C-API). The nearest
upstream commit is "Treat playedOn 'network' replays as console captures".
Our 09-22 request (`REPORT_from_exphil_2026-09-22_declared_profile_for_sceneless_replays.md`
in the sim repo) is an uncommitted note; nothing landed.

## Why ranked games diverge — measured, not guessed

Scoping method: run the sim's own validator (`tools/validation/validate_replay.py
--backend native --diagnostic`, CPU, not `mix`) on ranked FD games, as the
09-21 recipe says: if the validator fails, the gap is the sim's, not our
feeder. It did not even run — four format/declaration gaps, in order:

1. **No `scene.major`** in 2.0.1 game-start → `ValueError: replay start scene
   major is missing`. The runtime derives its whole capability set from it
   (`online_fnmsubs_zero`, `brawl_offscreen_damage`,
   `freeze_dead_up_fall_physics`, `whispy_dead_fighter_fix`); `wire.h` says
   the flags are explicit and independently selectable — only the
   *declaration path* is missing (the 09-22 request).
2. **No FrameStart event** (`frames.start.random_seed`, Slippi ≥ 2.2) → the
   per-frame RNG seed the runtime consumes is absent. 2.0.1 does carry the
   per-player pre-frame `random_seed`; the first present player's is a
   usable stand-in on FD (no stage RNG between frame start and the first
   fighter).
3. **No item frames** (Slippi ≥ 3.0) → item comparison impossible (lasers
   are items); fighters can still be compared strictly.
4. **Post-frame fields newer than 2.0.1** (`hurtbox_state`, `hitlag`,
   `misc_as`, `animation_index`, `last_attack_landed`, `combo_count`,
   `last_hit_by`, `velocities.*`) are required by the loader.

A throwaway experiment patch (`eval_runs/1006_sim_parity/validator_sceneless_experiment.patch`,
287 lines on `melee-sim-light` 330d0a3c, env-var driven: `MSL_DECLARED_SCENE_MAJOR`,
`MSL_UCF_*`, `MSL_FNMSUBS`) makes all four tolerant; absent fields read as
zero and are skipped by the comparison. With it the validator runs on
2.0.1 games. Then, as the oracle over the unknown 2020 build flags
(`eval_runs/1006_sim_parity/oracle_sweep.txt`, four FD games, matching
rows / total):

| profile | 13_02_35 Marth+Fox | 00_41_46 Falco+Fox | 00_51_07 Marth+Fox | 01_06_10 Falco+Fox |
|---|---|---|---|---|
| online code set, all UCF on (today's default) | 90 / 11,456 | 119 / 13,159 | 105 / 8,838 | 117 / 9,762 |
| + `ucf_cardinals_1_0 = 0` | 1,808 | 3,452 | 6,827 | 7,455 |
| + `ucf_shield_drop_084 = 0` | 1,808 | **10,522** | 6,827 | 7,455 |

`fnmsubs` retail vs dolphin-legacy, shield-SDI, SDI and extended shield
drop made no difference on these four. So the 2020 ranked build had **no
UCF 1.0 cardinal snapping** (and at least one game says no 0.84 shield
drop); with that declared, the best game matches 80 % of its rows with 33
fields ever differing.

**The remaining first mismatch is one event, the same in 4/4 games:** after
~120 exact frames (the whole countdown), at frames −3…0 a player's
`last_attack_landed` becomes 18 (neutral-B = laser) and the other takes
3 % in the sim — **the opening laser connects in the sim and whiffs in the
recording** (recorded: the victim is already in DASH / action 42). Later
mismatches in the 80 %-game are episodic (124 rows of `action_id[1]` over
13k) — the trajectories re-converge, which says the physics is right and
specific events are wrong, the signature of a missing/different patch
rather than a feeder error. Candidates for the laser: a Slippi-online-era
hitbox/hurtbox or projectile difference, a countdown actionability rule, or
an input-lane semantic (2.0.1 processed vs raw lanes) — the sim's own
mismatch tooling (first-writer diagnostics, `--start-frame`) is the way to
settle it, one game at a time, exactly as they pinned Mewtwo/G&W.

## The branch

Base: `kyhavlov/melee-sim-light` main (the runtime); our C-API work lives
on `msl-main` `exphil-replay-step` and rebases onto it.

1. **Declared code-set profile (upstreamable, zero cost when off).**
   Suite/manifest field + CLI `--code-set slippi-online-2020` → the `wire.h`
   flags (online set + cardinals off + shield-drop-0.84 off + the rest as
   the oracle settles them); keep the hard error when neither scene major
   nor a declaration exists, with the error naming the flag. Same
   declaration on `msl_batch_reinit` / `EnvBatch` config so a match seeded
   from a 2.0.1 replay runs with the validator's flags, and in
   `ExPhil.Sim.Seed` (pass the profile with the savestate). ~1 day.
2. **Pre-3.0 recording support in the validator (upstreamable as a
   "legacy recording" admission class).** FrameStart fallback to the
   pre-frame seed, item comparison skipped when absent, optional-era
   post-frame fields skipped — the experiment patch, done properly
   (presence flags on `ReplayPlayer` instead of globals; reported in the
   result as `compared_fields`). ~1 day. Whether upstream *admits* such
   recordings is their call (their admission rules want items + full
   fields); for us a diagnostic-only class is enough.
3. **Chase the opening laser** with their tooling on `00_41_46`
   (80 % match) — first-writer operands at frame −3, hitbox/hurtbox
   of both fighters, laser spawn/velocity vs the recorded pre-frame
   inputs. Unknown size; the kind of thing that is one patch once found.
   Then the next first-mismatch, until a game passes to the end; the
   corpus has ~3k FD Fox games to confirm on.
4. **Feeder check** (`Seed.from_replay` against 2.0.1 lanes) only after
   the validator passes a game — the 09-21 recipe.

Not upstreamable / ours only: nothing so far. Efficiency cost: none when
the profile is not declared (flags already exist in the runtime).

## What this buys

Expert labels on the bot's own states from the actual training
distribution (same players, same code set), which is the only lever that
attacks the silent fall's mechanism head-on (INPUT_COHERENCE "10-05
23:30", "10-06"). Also unblocks the human lane of the sim gate (step 11,
blocked since 09-19) and the coach review on ranked games.

## Reproduce the experiment

```
# scratch worktree of melee-sim-light at 330d0a3c, LFS smudge skipped
GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach WT HEAD
cd WT && git apply ~/git/exphil/eval_runs/1006_sim_parity/validator_sceneless_experiment.patch
ln -s ~/git/melee-sim-light/data data   # extracted game data
mkdir -p build/melee_core/native && ln -s ~/git/melee-sim-light/build/melee_core/native/melee-core-native build/melee_core/native/
make validator PY=~/git/melee-sim-light/.venv/bin/python
MSL_DECLARED_SCENE_MAJOR=8 MSL_UCF_CARDINALS=0 MSL_UCF_SHIELD_DROP_084=0 \
  ~/git/melee-sim-light/.venv/bin/python -m tools.validation.validate_replay \
  --backend native --no-build --diagnostic "<ranked FD .slp>"
```
