# Context-aware fit — warm-history recovery closed (2026-09-13 evening)

Follow-up to [RECORDED_CONTEXT_2026-09-13.md](RECORDED_CONTEXT_2026-09-13.md)
"Next bounded fitting protocol" 1-4. Root: `eval_runs/0913_context_fit/`.

## Candidate

`eval_runs/0913_context_fit/round21/candidate.bin`
SHA256 `8bd1b009a5504cdb7aa42c12c65e6f4d2de90aa294e096cac0ec319d2296426f`.
Contract unchanged: windowed GRU F32, zero state per window, GRU64 x2, AR
head, window 16, queue 3, previous action on, dropout 0.0, reaction delay 2.
Initialization = the prior proof's zero-state/F32 `initial.bin` (57 tensors
compared, max |diff| 0.0, `initial_comparison.log`); fresh optimizer.

Pool: canonical fixture + the 18 `clips_v6` clips (9 handoffs x cold/warm,
warm = 18-frame input-only prefix) = 10,641 supervised targets; first 18
supervised targets of each clip x64 -> 31,053 draws / 486 batches per epoch.
The drill printed exactly that budget.

## Two declared rounds

| round | epochs | updates | final loss | frozen gate | live |
|---|---:|---:|---:|---|---|
| round9 | 9 | 4,374 | 0.00708 | FAIL 13/18 | withheld (protocol) |
| round21 | 21 | 10,206 | 0.000348 | PASS 18/18 | below |

round9 was the handoff's suggested budget; its loss was still falling ~25%
per epoch and the prior proof (same recipe, smaller pool) had only reached
1e-4 at epoch 18, so a second round at the prior proof's epoch budget was
declared before launch (`round9/VERDICT.md`). round9 is retained as the
failed attempt; nothing was extended within a run and no snapshot was fished.

## Frozen fit, round21 (`fit.json`, `fit_gate.json`)

All 18 clips: 18/18 early conditional matches, minimum early joint target
probability >= 0.995 (most 1.000). Whole-clip argmax: canonical 99.986%,
the two long interruption clips 99.72% (one row each), everything else 100%.
Rows are indexed by SUPERVISED target (the input-only prefix is never a row).

Baseline for contrast — the prior candidate measured on the same 18 clips
(`baseline_round21_fit.json`): cold familiar 18/18 at >= 0.997; EVERY warm
recovery clip had a 0.000-probability early target (4_warm 15/18, 75_warm
13/18, 146_warm 14/18); interruption clips 9-12/18 cold and warm.

## Closed loop, round21 (reaction 2, T=1.0, libmelee AF, 2 runs/handoff)

| gate | result | prior candidate |
|---|---|---|
| warm familiar controls (committed history, 120f) | **12/12**, chains 13-14 | 6/12 (recovery 4/75/146: 1/0/0) |
| isolated interruptions 502/1389/3152, neutral opp, 360f | **6/6 strict recoveries**, 35-39 full cycles each, ready-to-cycle 10-12f, 0 deaths | 0/6 |
| interruptions, REPLAY opponent (the human keeps attacking), 360f | 6/6 recover at least once: chains 39/40, 13/14, 8/14; 3152 run 2 recovers 3 times (7 later hits) | 0/6 |
| cold familiar controls (cold history, 120f) | 10/12 x2 batches: everything 13-14 EXCEPT sustain-900 = chain 1,7 / 1,1 | 12/12 |

All runs have valid input timing, zero prefix drift, no truncation. The
teacher's own chains on the isolated interruptions were 40/37/38.

### The one regression: cold sustain-900

The mid-chain handoff at frame 900 with COLD (repeated-first-frame) history
now breaks at the first policy-issued frame: cold issues B where warm
issues B+X (the jump-cancel), one frame late, and the chain never forms
(`cold_controls_rerun.json` run 4/5 `policy_inputs`; warm run 4 for
contrast). Teacher-forced, `900_cold` is 18/18 at p=1.000 — so this is the
live cold seam (pending prefix commands + cold padding) landing between the
two histories the model now separates, not an unfit clip. The prior candidate
passed it 4/4; this one fails it 4/4. The cold contract is an artifact of the
earlier proof (real play always has real history: the warm contract is the
deploy one), but it is the earlier proof's regression gate, so it is recorded
as a fail, not waved through. Open decision for Bradley (below).

### Harness note

3 of 24 cold-history runs ended with `dolphin ended: dolphin_disconnected`
(always the first or last run of a batch); 0 of 36 warm/committed runs did.
Errored runs are excluded, not scored. Not investigated.

## Standing conclusions

1. Warm-history recovery was a COVERAGE gap, closed by training the same
   handoffs with their real 18-frame input-only prefix. No history reset,
   no architecture change, no label change on this line.
2. The frozen early gate (cold AND warm, supervised-indexed) predicted the
   live outcome on every warm/interruption case. It did not predict the
   cold-900 live failure: a perfect teacher-forced fit of a cold clip does
   not certify the live cold seam.
3. Budget: this recipe needs ~20 epochs on this pool; 9 is under-spent.

## Not done / open

- No default installed (`scripts/play_multishine_proof.sh` still plays the
  prior candidate). Bradley: either accept cold-900 as a known limitation of
  a contract nobody deploys and point the play script at round21, or require
  the cold gate too (then the next round needs a cold-seam-faithful clip).
- Only the nine trained handoffs were tested. Held-out starts/perturbations
  (different positions, facing, opponent characters) remain untested.
- `dolphin_disconnected` in cold mode.

## Human session (22:48): passed

Bradley played round21 locally and called it the best version yet. Strict
benchmark of the replay: max chain 101, 330 completed cycles, 209 ground
shines/min, 1 stock lost, 17/31 hit episodes recovered (typical ready-to-cycle
8-11 frames); the prior candidate's session was chain 27 / 78 cycles / 54 per
min / 4 stocks / 9 of 46. See HANDOFF_2026-09-13d addendum.
