# Mewtwo Ship Runbook

**Goal (the only one that counts):** a Mewtwo bot you can launch into Dolphin against a
level-9 CPU and *watch play a recognizable game of Melee*. Not a val-loss number — a recording.
That's the gate, and it's binary.

**Last updated:** 2026-07-04

> This replaces the old architecture-sweep plan, which was written against the deprecated
> `train_from_replays.exs` (with `--hidden-sizes`, `--temporal`, `--warmup-steps`). The current
> script is **`scripts/train.exs`**, which auto-applies temporal / precision / lr-schedule per
> backbone and takes `--num-layers` (there is no `--hidden-sizes`). Following the old commands
> verbatim errors on flag parsing. The old breadth-first sweep is preserved as an appendix at the
> bottom — but **do not run it until a bot ships.**

---

## The rule: depth on ONE config, all the way into Dolphin

The repo is a mature lab (30+ architectures, benchmark harness, self-play, PPO, ONNX). The one
thing it has never produced is a bot that plays. Everything below serves that and nothing else.

- **One character:** Mewtwo (129 replays; teleport/wavedash tech is the interesting part).
- **One backbone:** **GRU** first. It's top-5 on the benchmark, trivially 60fps, and the *fastest
  to iterate* — which is what matters when the loop is train → watch → fix. S4 is the "does the
  fancy one actually beat the classic" control, run **only** after GRU has shipped once.
- **Do NOT** run a second architecture, re-run the benchmark, or touch self-play until step 4
  produces a bot that plays. Breadth is the comfortable move; depth is the correct one.

---

## Prereqs

- Replays at `/workspace/replays/mewtwo` (RunPod) or `$EXPHIL_REPLAYS_DIR/mewtwo` (local).
- `mix compile --force` once before the first run (avoids stale-build SIGSEGV — see CLAUDE.md).
- **Never run `mix` while a training run is live** — recompiling EXLA's NIF kills the run.

---

## Step 1 — Train GRU to convergence (ONCE, properly)

Not a 3-epoch smoke test. A real run: 40 epochs, focal loss on (Mewtwo leans on rare buttons —
Z teleport-cancel, L/R wavedash/tech), early stopping, save-best.

```bash
mix run scripts/train.exs \
  --backbone gru \
  --replays /workspace/replays/mewtwo \
  --train-character mewtwo \
  --epochs 40 \
  --batch-size 128 \
  --num-layers 2 \
  --focal-loss --focal-gamma 2.0 \
  --label-smoothing 0.1 \
  --early-stopping --patience 8 \
  --save-best \
  --seed 42 \
  --name gru_mewtwo_v1
```

Backbone defaults (temporal on, bf16, cosine schedule, window size) are applied automatically —
don't pass them. Run `mix run scripts/train.exs --help` to see every flag.

**Watch the per-epoch diagnostics** (the `Diagnostics` callback prints them). The one that matters:

```
⚠ COLLAPSE WARNING (batch 500): diversity=1, max button prob=0.98
```

If you see that, the model has mode-collapsed to one action — stop and go to the playbook
(`docs/guides/BC_FAILURE_MODES.md`) before burning 40 epochs. Also eyeball the per-button
predicted-vs-actual press rates: if predicted ≫ actual on A/L/R, focal loss needs tuning.

**What you ship:** `checkpoints/gru_mewtwo_v1_best.axon` + `gru_mewtwo_v1_best_policy.bin` +
`gru_mewtwo_v1_config.json`. The `_best_policy.bin` is exported from the **best-by-val** weights
(via the Checkpoint callback), *not* the final epoch — so `--save-best --early-stopping` ships your
best model, not your second-best. Always play the `_best_policy.bin`.

---

## Step 2 — Eval the checkpoint (sanity, not celebration)

```bash
mix run scripts/eval_model.exs \
  --policy checkpoints/gru_mewtwo_v1_best_policy.bin \
  --replays replays/mewtwo \
  --detailed
```

`eval_model` auto-loads `gru_mewtwo_v1_config.json` (it maps `_best_policy.bin` → `_config.json`),
so you don't re-specify architecture flags. Confirm val loss matches training and the detailed
per-head breakdown isn't degenerate (buttons head near-constant = collapse). A good number here is
necessary but **not** sufficient — a bot with great val loss can still crouch in the corner.

**Correctness gate between eval and Dolphin:** the overfit-replication test
(`mix test test/exphil/training/overfit_replication_test.exs --include slow`) proves the
train→embed→discretize→decode pipeline can memorize and reproduce a known behavior
(Fox multishine, ~30s on GPU). If Step 3 looks broken, run this FIRST — it separates
"pipeline bug" from "model isn't good enough" categorically. A real recorded fixture
also exists at `test/fixtures/replays/fox_multishine.slp` (made by
`scripts/record_multishine.exs`) for Tier-2/Peppi-path checks.

---

## Step 3 — Drive it in Dolphin (the moment of truth)

This is the step the whole repo has been avoiding. Do it early and often.

```bash
mix run scripts/play_dolphin_async.exs \
  --policy checkpoints/gru_mewtwo_v1_best_policy.bin \
  --dolphin /path/to/dolphin \
  --iso /path/to/melee.iso
```

Set the opponent to a level-9 CPU. **Watch the frames.** You are not looking for wins — you are
diagnosing *which* failure mode you have. See `docs/guides/BC_FAILURE_MODES.md` for the full
symptom → cause → fix table. The fast triage:

| What you see | Likely cause |
|---|---|
| Freezes / holds one input | Mode collapse (loss/data) |
| Twitchy, never commits | Stick discretization / no temporal smoothing |
| Plays but SDs off-stage every time | Recovery unseen in data / no reward signal (BC ceiling) |
| Reasonable neutral, whiffs punishes | Data volume — expected at 129 replays |

---

## Step 4 — Fix the highest-leverage thing, retrain, loop (max 3–4 times)

Change **one** thing per loop, retrain step 1, re-watch step 3. Do not tune five knobs at once —
you won't know what helped. When it plays a recognizable game: **record it. That's the ship.**

Only *after* that recording exists do the follow-ups become worth anything:
- Run S4 with the identical config as the "fancy vs classic" control.
- Longer run / more replays (see data expectations below).
- ONNX INT8 export for 0.55ms inference (`docs/guides/INFERENCE.md`).
- *Then* consider self-play / RL to push past the BC imitation ceiling.

---

## Data expectations (set them honestly)

| Replay count | Realistic outcome |
|---|---|
| 129 (current) | Basic movement, some combos, inconsistent neutral, shaky recovery |
| 500+ | Solid neutral, better recovery decisions |
| 2000+ | Matchup-aware, consistent |

At 129 replays the ceiling is "recognizably plays Melee," not "tournament-viable." That's fine —
recognizably-plays is the ship gate. More replays is a post-ship lever, not a blocker.

---

## RunPod → local handoff

```bash
# On RunPod after training
sync-all-up
# On local machine
rclone copy b2:your-artifacts-bucket/checkpoints/ ./checkpoints/ --progress
```

Sync the whole trio — `_best.axon`, `_best_policy.bin`, `_config.json` — or eval/play can't
reconstruct the model.

---

## Appendix — Architecture sweep (POST-SHIP ONLY)

The former Phase 2–4 breadth sweep (MLP / LSTM / GRU / Mamba / sliding_window / Jamba, plus
hyperparameter grids) lives in git history. It is **deliberately not reproduced here** because
running it before a bot ships is the exact trap that has kept this project in lab-mode. Once a GRU
bot plays in Dolphin and you have a task where a 0.1 val-loss delta actually *means* something,
revive the sweep with the `train.exs` interface: swap `--backbone gru` for the target, keep
everything else identical, and give each run a distinct `--name`.

## 2026-07-07 addendum: data facts that change the plan

Per-character data visibility (now logged every run) revealed the general
200-file corpus is fox 25% / falcon 23% / falco 19% / marth 10% / sheik 5% —
**Mewtwo does not appear in the top 8**. The general model has essentially
never seen Mewtwo piloted. Implications:
- The specialist run on `replays/mewtwo` (131 files) is not a nice-to-have,
  it is the ONLY source of Mewtwo piloting signal.
- Consider general-pretrain → Mewtwo fine-tune rather than from-scratch.
- Calibration from the reference project: slippi-ai's strong BC used ~100k
  replays and a ~20M-param model (we: 200 replays, 3.7M params). Weak main-
  stick accuracy (mx≈35%) at our scale is expected, not a bug — scale data
  before blaming architecture.
- Loss fixes landed 2026-07-07 (GOTCHAS #53/#54): retrain anything trained
  before that date before drawing conclusions from its play.
