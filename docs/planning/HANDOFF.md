# Session Handoff — 2026-07-04

> For the next Claude session (and future-me at the keyboard). Written during a
> brainstorm session done mostly from a phone; the user is now resuming on a real
> keyboard at home with GPU access. All work is on branch
> **`claude/project-direction-brainstorm-dtj58b`** in exphil, PHMUB, and edifice.

## The one-line state

**Priority #1 is shipping a bot** (a Mewtwo BC agent that visibly plays Melee in
Dolphin), not adding lab features. PHMUB (the genetics game) is the *next* project
after exphil's bot ships. edifice/nx work is supporting infrastructure, not a
headline. The active in-flight task is an **overfit-replication test harness**
(Vlad Firoiu's advice) — half-built, needs the training wiring closed at a machine.

## What this session decided / did

**Framing (agreed with user):** exphil is a mature *lab* that has never produced a
*bot that plays*. The whole apparatus (30+ architectures, benchmarks, self-play,
PPO, ONNX) exists to serve one deferred, binary deliverable: a recording of a bot
playing recognizable Melee. Depth on ONE config into Dolphin > breadth.

**Committed this session (exphil):**
- `lib/exphil/training/callbacks/checkpoint.ex` — export a runnable
  `_best_policy.bin` from the **best-by-val** weights whenever a new best `.axon`
  is saved. Fixes a real footgun: previously the only `_policy.bin` was written at
  train-end from *final-epoch* weights, so `--save-best --early-stopping` silently
  shipped the second-best model. **Not yet verified with a real run** — confirm the
  `_best_policy.bin` appears and loads in Dolphin.
- `docs/planning/MEWTWO_TRAINING_PLAN.md` — rewritten as a tight **ship runbook**
  against the current `scripts/train.exs` interface (the old plan used the
  deprecated `train_from_replays.exs` flags like `--hidden-sizes` that error). GRU
  first, one config, straight into Dolphin. Arch sweep demoted to a post-ship appendix.
- `docs/guides/BC_FAILURE_MODES.md` — Dolphin symptom → cause → real-lever playbook
  (mode collapse, twitch, recovery ceiling, covariate shift, netplay lag).
- `CLAUDE.md` — relaxed the blanket "never run mix" rule: it only applies when a GPU
  training run is live on the dev box. **In cloud/CI/idle, run `mix` and the tests
  freely** — that's the point of them.

**In-flight this session (exphil) — the overfit-replication harness:**
Vlad Firoiu (slippi-ai/Phillip author) advised: train on a short replay of a
specific behavior (e.g. multishining) and check the bot can replicate it — this
collapses the "is it underfitting or a bug?" ambiguity that cost Phillip ~2 years.
A memorized behavior a correct pipeline can *always* reproduce; failure to reproduce
= categorically a pipeline bug (embedding / discretize / decode / sampling), which
is exactly the class of the cache-poisoning bug just fixed.

- ✅ **Piece 1 — synthetic fixture.** `ExPhil.Test.ReplayFixtures.tech_fixture(:multishine, opts)`
  returns `{%GameState{}, %ControllerState{}}` frames with Fox shining on a fixed
  period. **Critical property baked in:** the *state varies with shine phase* (Fox's
  `action`/`y` cycle), because a periodic output cannot be reproduced from constant
  input — a constant-state fixture would fail a *correct* pipeline.
- ✅ **Piece 2 — the checker.** `ExPhil.Test.ReplicationCheck.check/3`, a pure
  function with three strictness levels: `:exact` (frame-for-frame, buttons + stick
  bucket), `:periodic` (default; shine events recur at the right period ± tolerance,
  count ± tolerance), `:loose` (shine rate ± tolerance). Returns a rich diagnostic
  map, not a bare bool. Shine = **B held + main stick down** (fixture and checker
  agree by construction).
- ✅ **Piece 2 test.** `test/exphil/replication_check_test.exs` — pure, fast, no GPU.
  **First thing to run at the keyboard: `mix test test/exphil/replication_check_test.exs`**
  to confirm pieces 1 & 2 compile and pass. (Written blind — could have a field-name
  typo; fix any and it's solid.)
- ⏳ **Piece 3 — the training test (NOT built; the two seams need a running instance).**
  See skeleton below.

## Piece 3 — training wiring skeleton (close this at the keyboard)

Goal: train a small model to memorize `tech_fixture(:multishine)`, decode its
per-frame predictions to `%ControllerState{}`, run `ReplicationCheck.check/3`.

Two seams I couldn't verify without compiling/running — resolve these first:

1. **Ingest a synthetic fixture into training.** `ExPhil.Training.Data.from_frames/2`
   exists (`lib/exphil/training/data.ex:143`) and builds a dataset from in-memory
   frames — but confirm the exact **frame map shape** it expects (game_state +
   action/controller keys, name_id). The existing quality tests
   (`test/exphil/training/training_quality_test.exs`) train via `Pipeline.setup!`
   off a replay *dir*; you either (a) adapt `from_frames` into a Trainer, or (b)
   write the fixture out as a `.parsed`/`.term` file (that's what `list_replay_files`
   loads — `.parsed/.bin/.term`) into a temp dir and point `--replays` at it. Option
   (b) is likely the least-friction path and also exercises the real load path.

2. **Decode predictions → controller.** `predict_fn` returns the 6-head tuple
   `{btn_logits, mx_logits, my_logits, cx_logits, cy_logits, sh_logits}` (see the
   diversity test in `training_quality_test.exs`). Need the inverse of the training
   discretization: `argmax` each stick head → bucket index → 0..1 stick value
   (`ExPhil.Embeddings.Controller` holds the bucketing; find/derive the
   bucket→value fn), sigmoid+threshold buttons → booleans, assemble a
   `%ControllerState{}`. Use **deterministic/argmax** sampling so `:exact` is meaningful.

Sketch:
```elixir
# test/exphil/training/overfit_replication_test.exs   @moduletag :slow, :gpu
frames = tech_fixture(:multishine, frames: 128, period: 8)
# 1) train to memorize: no val split, no reg, many epochs, tiny model, high LR
#    (write frames -> temp .term dir, or Data.from_frames -> Trainer)
# 2) predict per-frame, decode -> [%ControllerState{}] (seam 2)
# 3) expected = Enum.map(frames, &elem(&1, 1))
assert {:ok, _} = ReplicationCheck.check(expected, emitted, strictness: :periodic)
# then tighten to :exact once :periodic is green
```
Where it slots in the ship plan: it's the **correctness gate between eval (Step 2)
and Dolphin (Step 3)** in MEWTWO_TRAINING_PLAN.md — prove the pipeline can memorize
before asking whether it can generalize. Add that cross-link once piece 3 lands.

## Tier-2: a REAL multishine `.slp` (optional, higher-fidelity)

The synthetic fixture catches ~all pipeline bugs; a real `.slp` also tests the Peppi
parse path. How to make one:
- **Easiest:** Slippi Dolphin auto-saves every game to `.slp` (replay saving on by
  default; path in launcher settings). Vs mode, Fox on a known port vs a lvl-1 CPU,
  flat legal stage, multishine ~10–20s, end game → newest file in the Slippi dir. It
  need not be frame-perfect.
- **Reproducible:** script it via the existing libmelee bridge (`priv/python`,
  `MeleePort`, `play_dolphin` scripts) — issue the shine loop each frame with Slippi
  recording on; the input script becomes the ground truth.
- **Don't trim the `.slp` binary** (fiddly UBJSON) — record longer and **slice the
  frame range in the loader** (filter Fox's shine action-state). Put it at
  `test/fixtures/replays/fox_multishine.slp` and skip the Tier-2 test if absent.

## Cross-repo snapshot

- **exphil** — active. Ship a Mewtwo GRU bot (see MEWTWO_TRAINING_PLAN.md). Finish
  the overfit harness (piece 3). Verify the best-policy-export fix with a real run.
- **PHMUB** (genetics game) — planning-complete, **zero code**, the *next* project.
  This session ratified the Phase-0 minigame boundary in `docs/DECISIONS.md` (the
  fun gate is **M3 Gate-a-Cross on the M1 engine**, monohybrid red-hair 3:1 — NOT the
  Spore cell game; Loschbour×Stuttgart recombination deferred) and added
  `docs/PHASE0-CHECKLIST.md` (one-screen Steps 0–5 for the keyboard). **Open design
  question the user flagged:** "gate-a-cross isn't fun *yet* but could be" — needs a
  concrete hypothesis of *what* makes the cross fun (the gamble? stacking favorable
  alleles? watching a population shift?) before Step 4 is built. That sentence should
  drive Step 4's design.
- **edifice** — "done" (all roadmap tiers complete, on Hex). Maintenance/harvest mode.
- **nx fork** — real upstreamable work sitting unmerged: CallbackServer leak fix
  (PR #1682), configurable GPU allocator, fused selective-scan XLA custom call
  (branch `feat/edifice-lazy-callback-allocator`). Supporting infra for faster
  training; land opportunistically. Open call: is the custom-call pattern something
  nx maintainers want in-tree vs an external package.

## Immediate next actions (keyboard, in order)

1. `mix test test/exphil/replication_check_test.exs` — confirm pieces 1 & 2 (fix any
   blind typos). Fast, no GPU.
2. Close piece 3's two seams (ingest + decode), land the overfit test, run it. If a
   memorized multishine can't be replicated → you have a live pipeline bug; that's a
   win, chase it.
3. Verify the best-policy-export fix: a short `--save-best --early-stopping` run
   produces a loadable `_best_policy.bin`.
4. Then the actual ship: MEWTWO_TRAINING_PLAN.md Step 1 (GRU, 40 epochs, focal loss)
   → eval → Dolphin. Watch frames; use BC_FAILURE_MODES.md to diagnose.
