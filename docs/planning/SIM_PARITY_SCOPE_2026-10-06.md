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

### Branch opened 10-06 22:40 — `exphil-parity` (Bradley's go)

Worktree `~/git/msl-parity` (branch `exphil-parity` off `origin/main`
`45bae153`, in the `~/git/melee-sim-light` repo; `data` and `.venv` are
symlinks to the main clone; LFS fixtures materialised with `git lfs
checkout` inside the sim's `devenv shell`, which is also where `git` and
`python` run there). **Items 1 + 2 landed in one commit, `2b467dea`**, and
smaller than scoped: the 2020 online flags were already declarable through
existing knobs (`playedOn` metadata sets the three Dolphin call-site
patches; `fnmsubs_profile="dolphin-legacy"` sets the zero-sign one), so
no new "code set" concept was needed. What changed:

- `parse_start` records `scene_major_missing` instead of failing; a
  sceneless recording then *requires* `fnmsubs_profile` (the error names
  it). Modern recordings are untouched.
- Sceneless only: FrameStart seed → first present fighter's pre-frame
  seed (`frame_seed_at`), seed lane not compared; no item frames → item
  compares skipped; absent post-frame columns (`hurtbox_state`, `hitlag`,
  `animation_index`, the five velocities) read as zeros and their specs
  (leader + follower) are skipped. Per-replay state on `ReplayView`, no
  globals. Result gains `skipped_fields` + `scene_major_missing`; the CLI
  prints "legacy recording (no scene major): not compared: …".
- CLI `--[no-]ucf-cardinals-1-0`, `--[no-]ucf-shield-drop-084` (+ the
  other three) for manual/diagnostic runs.
- Validator tests: 9 passed, 1 skipped (PPC artifacts).

Baseline on the branch (`--diagnostic --fnmsubs-profile dolphin-legacy
--no-ucf-cardinals-1-0 --no-ucf-shield-drop-084`, 4 ranked FD games):

| game | matching rows | first mismatch |
|---|---|---|
| 00_41_46 [SM] Falco + Fox | **10,665 / 13,159** (exp. patch on old base: 10,522) | f −1: Fox takes 3 % (laser lands in sim) |
| 00_51_07 [RUDE] Marth + Fox | 6,827 / 8,838 | f −2: Marth takes 3 %, Fox `last_attack_landed=18` |
| 01_06_10 [=3] Falco + [JAKE] Fox | 7,479 / 9,762 | f 0: Fox takes 3 % |
| 13_02_35 Marth + Fox | 1,808 / 11,456 | f 6: `shield_hp[1]` 59.16 vs 59.23; f 27 the laser |

So item 3 is exactly one event in 4/4 games — the opening laser connects
in the sim and whiffs in the recording (frames −2..0 in three, frame 27 in
`13_02_35`). The 4th game's earlier `shield_hp` lead is benign: sliding
`start_frame` 2..18 shows both sides REGENERATING at +0.07/frame with the
sim exactly one frame ahead (sim f6 = recording f7), converging at 60 by
f17 — a one-frame-earlier shield release, i.e. the analog-trigger release
frame crossing the shield threshold differently under `trigger_u8`'s
rounding of the 2.0 physical L/R lane; self-healing, low priority.

Laser chase plan (needs sim-side item visibility, since 2.0 recordings
have no item frames): dump the sim's laser spawn position/velocity and
both fighters' positions/actions for frames −10..0 (runner trace →
`tools/viewer/msltrace1.js`, or a `--diagnostic` detail dump), and compare
the fighters' lanes with the recording — positions match to the frame
before the hit in all four games, so the candidates are the laser's spawn
offset/velocity on the 2020 build, or the dashing Fox's hurtbox placement. Admission unchanged: these run as `--diagnostic` until a game
passes and gets a provenance record. Still to do on the branch: the laser
chase (item 3), the `msl_batch_reinit`/`EnvBatch`/`Seed` declaration
(rest of item 1), a sceneless-fixture test for upstreaming.

### 10-07 results — 3 of 4 games bit-exact; the corpus is 2019 console

Sim commit `0dc7d4a5` on `exphil-parity`. Same four games, same flags
plus `MSL_POST_FRAME_MAP_PASS=1`:

| game | 10-06 baseline | 10-07 |
|---|---|---|
| 00_41_46 [SM] Falco + Fox | 10,665 / 13,159 | **13,159 / 13,159** |
| 01_06_10 [=3] Falco + [JAKE] Fox | 7,479 / 9,762 | **9,762 / 9,762** |
| 13_02_35 Marth + Fox | 1,808 / 11,456 | **11,456 / 11,456** |
| 00_51_07 [RUDE] Marth + Fox | 6,827 / 8,838 | 8,796 / 8,838 (42 rows: `action_id[1]` 88 vs 91, f4681–4722, reconverges) |

**The corpus is not 2020 online.** `peppi` metadata: `playedOn:
nintendont`, `startAt` 2019-05-14 / 05-21 / 09-28, Slippi *recorder*
version 2.0.1. These are console tournament captures — which is why
`fnmsubs` made no difference and why UCF 1.0 cardinals / 0.84 shield
drop are off (UCF 0.73 era). The `dolphin-legacy` profile is only the
declaration key the validator needs for a sceneless file; the arithmetic
is retail's.

**Defect A — the recorder hook moved (2020-06-06).** slippi-ssbm-asm
`9398d52` "move post frame back, was missing some data changes" moved
`SendGamePostFrame` from GALE01 `0x8006C5D8` (epilogue of
`Fighter_procMap`, proc priority 6 — before `Fighter_ProcessHit`) to
`0x8006DA34` (`Fighter_UnkCallCameraCallback`, after the hit pass). The
sim models the later hook. So in 2019 recordings every hit's
bookkeeping (percent, last_attack_landed, combo, last_hit_by, hitstun,
shield_hp regen) appears one frame later than the sim's sample while
positions already agree — the "opening laser whiffs in the recording"
of 10-06 was this: `MSL_DUMP_FRAMES=-8:1` shows the laser (sim item type
54, vx −7) reaching Marth at f−2 in the sim and the recording taking the
3 % at f−1. Hosted builds can now snapshot each fighter at the procMap
epilogue (`msl_slippi_post_frame_map_pass_sample`, shadow `Fighter`
copies outside `MslCoreMatch`); `write_compare` reads the snapshot and
its `cur_pos`. Effect on 00_51_07 alone: 2,011 mismatching rows → 42.

**Defect B — mine: the frame-start RNG restore.** The runtime restores
the HSD seed twice per validation frame: at frame start (modern
`FrameStart` seed) and at the first fighter's pre proc (per-player
pre-frame seed). My 10-06 fallback fed the pre-frame seed into the
frame-START slot for sceneless files, so everything scheduled before the
fighter-pre proc — `Fighter_8006A1BC` (priority 0: hitlag decrement →
`Fighter_8006D10C` → `ftCo_8008DCE0`, whose `HSD_Randf() < x240` picks
DamageFlyRoll over DamageFlyN), script GFX jitter at priority 1–2 — ran
from a state the source never had there. `MslCoreStageEvents` gains
`frame_random_seed_missing`; `msl_core_match_step_begin` skips the
frame-start restore when set (wire sizes 133 / 189). Validator tests 9
passed. With both fixes, three games pass end to end on every compared
lane (positions, actions, shields, percents, hits, stocks).

**Residual (00_51_07 f4681).** The one event left is the same Roll pick:
without a FrameStart seed the sim's state at priority 0 is carried from
the previous frame, where the headless effect model's RNG consumption is
partial by design (`MSL_SEED_DRIFT=1` reports a differing restore on
most frames — that is the model's incompleteness, absorbed by the
restore). Modern recordings are immune (FrameStart seed = retail's state
at priority 0). Closing it would need the effect model exact for the
frame preceding a hitlag-end launch — not worth it for seeding; cosmetic
(animation id only, trajectory identical).

**Dead ends recorded so they are not re-walked.** (1) A "2019 build skips
scripted GFX jitter / hit-SFX random" model (`MSL_SCRIPT_GFX_OFF`) made
2/4 games pass and was an artifact of defect B — the drift it "removed"
was the sim's own early-frame draws being double-counted; reverted.
(2) The runner is a `subprocess.Popen(stderr=PIPE)` read only at
shutdown: runtime-side `fprintf(stderr)` probes vanish — use
`MSL_PROBE_LOG=<path>` (`msl_probe_printf`). (3) `make native` is needed
for runtime edits; `make validator` only rebuilds `native.c`.

**Instruments kept:** `MSL_DUMP_FRAMES=lo:hi` (recorded vs sim fighter
lanes, per-player recorded seeds vs sim frame seed, every live sim item),
`MSL_SEED_DRIFT=1`, `MSL_PROBE_LOG`, `--played-on` (anonymized ranked
dumps lack `playedOn`), full `mismatch_fields` census under
`--diagnostic`. Removed after use: RNG backtrace tracer, GFX dispatch
trace, Roll/SFX probes (recipes in the session log if ever needed).

**Next on the branch:** promote `post_frame_map_pass` and the sceneless
declaration to `MslCoreMatchConfig` wire fields (≈15 mirrors: wire.c/h,
api.c, validate_replay.py, suite_io.py, viewer schema, tests), derive
them from the recording (recorder version < 3.0 ⇒ map-pass hook, no
FrameStart ⇒ carry RNG), carry them into `msl_batch_reinit`/`EnvBatch`/
`ExPhil.Sim.Seed`; then the feeder check (`Seed.from_replay` on 2.0.1
lanes) on the three passing games; then the corpus sweep (~3k FD Fox
games) for the pass rate. Upstream PR (hook + frame-seed fixes are
general) only with Bradley.

### 10-07 15:00 — corpus sweep: 381 / 1,245 FD Fox games bit-exact

`eval_runs/1007_sim_parity/` (`sweep_v2.log`, `sweep_v3_ucfoff.log`,
`sweep_v3_default.log`; game lists `fd_fox_v2.txt` / `fd_fox_v3.txt`).
The ranked FD Fox corpus is two console eras: **1,122 games at recorder
2.0.1 (2019)** and **123 at 3.9.0 (2021)**, all `playedOn: nintendont`
(5 say `network`). Sim at `7749e749` (recorder hook derived from the
format version; era lanes optional below 3.16).

| set | declaration | end to end | with mismatches | error |
|---|---|---|---|---|
| 2.0.1 (1,122) | dolphin-legacy key, cardinals off, 0.84 drop off | **313** | 807 | 2 (runner broken pipe) |
| 3.9.0 (123) | cardinals off, 0.84 drop off | **68** | 55 | 0 |
| 3.9.0 (123) | suite defaults (UCF 1.0 on) | 2 | 121 | 0 |

So the 2021 console build also had no UCF 1.0 cardinals / 0.84 shield
drop. Of the 807 non-passing 2.0.1 games: exact prefix q25/50/75 =
1,561 / 3,042 / 5,498 frames; 355 reconverge to an exact suffix; 275
have ≤ 100 mismatching rows (cosmetic episodes like the tumble pick);
the first mismatching field is `action_id` (400) or `pos_x` (281) —
events, not physics drift. Per opponent (pass/fail): Falco 164/329,
Marth 101/345, Sheik 48/120, Zelda 0/15. Sweep cost: 30 s wall for
1,122 games at 8 workers (it starves a concurrent trainer's CPU data
pipeline — 23 → 172 ms/it — so sweep between queues).

For seeding this is already broad: 31 % of games are exact for their
whole length and three quarters of the rest for ≥ 1.5k frames; the
next first-mismatch classes (`action_id` at f1k–5k) are the next chase,
one game at a time, same method (`MSL_DUMP_FRAMES`, seed-step counts).

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
