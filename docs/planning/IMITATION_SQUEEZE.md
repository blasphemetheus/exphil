# Imitation squeeze — pilot-first program for the generalist Fox (V3.2+)

**Written 2026-09-20.** Living doc: the plan is at the top, the status
ledger at the bottom. Rule from Bradley: **no long run until the pilots
have picked the ideas.** Every idea gets a small, matched pilot; the long
run (V3.2) is assembled from the pilot winners.

Reference point: **V3.1-ep3** (`checkpoints/fox_v3_1_20260919_022008_ep3/`),
genuine held-out anon 2.127 / registry 2.066 (GOTCHA #122 split), trainer
val 2.158, skips per pass 10 → 41 → 188 → 284. Recipe in GOALS.md Track D.

## Why pilots, and what a pilot can and cannot see

The full run is ~6.3 h per pass over 28,436 files; four passes cost a day
and the late-run behaviour (skip escalation, ep4 regression) only shows
after ~3 passes. Two pilot shapes cover the two kinds of question:

| Shape | What it answers | Cost | Instrument |
| --- | --- | --- | --- |
| **P-fit**: fresh or warm fit on a fixed 5 % subset (≈1,400 files, matched seeds 905/906), 3 passes | does the idea change what the model *learns* (held-out CE, identity liveness, D3 tell) | ≈ 1 h | `heldout.exs` with `HELDOUT_CORPUS=full` on the SAME 16-file validation split; effect must beat the seed-pair spread |
| **P-cliff**: resume from a late checkpoint (`…_ep3/model_epoch1.axon` = end of V3.1 epoch 3) with the lever applied, 3,000 batches over the known hard region | does the idea change *stability* (skip density on the same hard batches, ‖W_h‖ trend, saturation) | ≈ 20 min | trainer skip counter, `scratchpad/ckpt_audit.exs` norm audit, fatal-batch replay (`scripts/replay_fatal_batch.exs`) on the captured V3.1 batches |

What pilots cannot see: gains that need the whole corpus (identity slots
with < 5 games in the subset, rare situations). Those ideas go straight
into the long run as "low-risk carry" rather than being pilot-gated.

Pilot hygiene (all standing rules apply):
- The subset is hashed once and stored (`eval_runs/0920_squeeze/subset.json`
  with sha256s); every pilot uses it. Train-overlap with the validation
  split is printed and must be 0 (GOTCHA #122).
- Two seeds per arm minimum; report the pair spread next to the effect.
- Same deploy contract as V3.1 (F32, highest arithmetic, AR head, T=1.0);
  never change two things in one arm.
- A pilot that changes the loss surface (weights, smoothing) is judged on
  the **plain-CE** held-out protocol, never on its own training loss.
- No `mix` on the box while a pilot beam is live; pilots run as systemd
  units with the R6 exit-status fix landed first (see below).

## Ideas, ranked by expected value / cost

### Stability levers (P-cliff first, then carry into P-fit)

1. **Unroll 40** (`--unroll 40`, overlap 0). Halves the backward horizon
   that overflowed in V3; halves the per-step compute. Risk: shorter
   credit horizon for slow setups (edgeguards). Expected: skip density ↓,
   held-out flat or slightly worse. *Highest priority; cheapest.*
2. **Per-timestep gradient value clipping** (clip each element, before the
   global-norm clip). Keeps the global-norm scale finite when one timestep
   blows up, so the batch contributes instead of being skipped. Needs a
   trainer change (elementwise clamp in `train_step_bptt`). Expected: skip
   density → ~0 with the same held-out.
3. **Stronger weight decay (0.2)** on the GRU hidden kernels only
   (parameter-group decay). Norms fell 57.6 → 53.0 under 0.05; skips still
   rose, so this is second-order. Cheap to pilot on the cliff.
4. **Dropout 0** on the recurrent carry (keep input dropout). Dropout was
   the amplifier in the V3 diagnosis. P-cliff only.
5. **Hidden-kernel spectral clamp** (project ‖W_h‖₂ ≤ c after each step).
   Heavier, only if 1–3 fail.

### Learning levers (P-fit)

6. **V3.2 tag map**: apply `data/identity/entity_aliases.tsv` merges,
   void FOX/LI collisions, rerun identify + cluster, and cluster the
   anonymous 42 % with a lower bar for extra pseudo-slots. Judged on
   identity liveness (registry − anon gap) and D3 at n=10. Also the
   "low-risk carry" if the pilot is neutral.
7. **Corpus dedup** (33 % duplicates measured 09-09): drop exact
   duplicates and CPU-port games. Fewer files, same information; pilots
   will not see the gain, so this is a carry.
8. **More passes at lower peak LR** (cosine over 6–8 passes, peak 6e-5).
   ep4 regressed at the LR tail with 284 skips; if 1–2 fix the skips,
   longer schedules become worth it. P-fit with 6 passes on the subset.
9. **Width 768** (`--hidden-sizes 768,768,256`). Architecture is
   explicitly not the bottleneck (GOALS.md), so this is last among the
   learning levers; only after 1–2 land, and only if the P-fit curve is
   still descending at 3 passes.
10. **EMA of weights** for the exported policy (trainer has model EMA).
    Free to try on any P-fit: export both and score both.

### Behaviour levers (measured with fingerprints, not CE)

11. **Execution gap (D4)**: L-cancel press offset, short-hop rate, and
    wavedash rate sit below every probed human on every arm. Hypothesis:
    sampling at T=1.0 spreads the press timing; test `--buttons-temperature
    0.7` on the live rung only (a decode instrument, not a training change,
    per the no-bespoke-decode-rules feedback) to see whether the gap is in
    the weights or in the sampling. If it is in the weights, the training
    lever is the frame-timing loss weight (`--stick-edge-weight` analogue
    for buttons), piloted with P-fit.
12. **Pressure behaviours**: roll and spotdodge rates 3–6× the humans on
    every arm. Likely a distribution-shift symptom (bot gets hit more).
    This is the RL doc's problem, not imitation's; record only.

## The long run (V3.2) — assembled, not guessed

Launch criteria: at least one stability lever with skip density < 0.05 %
on the cliff AND held-out not worse than V3.1-ep3 on P-fit; the V3.2 tag
map built; dedup applied; R6 exit-status fix landed; the fatal-batch
captures from V3.1 replayed under the chosen lever. Warm start from
V3.1-ep3 or fresh is itself a pilot question (P-fit both).

Gates on the result are Track D's: D2 monotone per pass on the genuine
split, D3 at n=10, D4 measured, then D5.

## Prerequisite engineering (small, do first)

- [ ] R6 fix: `scripts/train.exs` runs the fit under `Task.await` or halts
      on abnormal `:DOWN` so a trainer raise fails the unit (found 09-20:
      raise in a spawned process exits 0).
- [ ] Per-epoch ‖W_h‖ + skip-density line in the trainer log (the cliff
      instrument, so no separate audit run).
- [ ] `eval_runs/0920_squeeze/` scaffold: `subset.json`, `run_pilot.sh`
      (arm name → unit), `score_pilot.sh` (held-out full split + identity
      liveness + fingerprint on 3 headless games).
- [ ] Per-timestep value clip flag (`--grad-value-clip X`), tested with the
      existing non-finite-grad tests.

## Status ledger

| Date | Pilot | Arm | Result | Decision |
| --- | --- | --- | --- | --- |
| 2026-09-20 | — | — | Doc written; V3.1-ep3 is the reference; no pilots run | Build the scaffold + R6 fix first |
