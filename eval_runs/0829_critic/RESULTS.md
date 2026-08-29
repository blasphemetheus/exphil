# RESULTS — D2 critic + Best-of-N, first run (shakedown) 2026-08-29

Design + pre-registered rule: `docs/planning/CRITIC_D2_DESIGN.md`. Read
it first. Everything here is the first run after two shakedown bugs
(`match?/3` vs `Kernel.match?/2`; `Activations` missing
`stage_internals` — GOTCHA #105). Policy `fox_gen_v1_..._ep10.bin`.

## Data

`critic_extract`: 40 erickfm FOX replays (sorted, files 1–40), port 1,
stride 2, K=8 samples at T=0.5 → **240,945 rows, 90,503 decision rows,
0 skipped**; φ = 574 (trunk 512 ⊕ raw 62). `cache/critic/fox_gen_v1_ep10_erickfm40.nx` (673 MB).

## 1. Held-out by replay (30 train / 10 eval, decision rows) — `train.md`

| metric | value | control |
|---|---|---|
| V pair-rank acc (chance 0.5) | 0.599 | shuffled target 0.514 |
| V held-out R² | 0.046 | |
| S top-1 among K+1 incl. master (chance 11.1%) | 49.8% | |
| sampling pass@1 | 16.0% | |
| **selector pass@1** | **22.2%** | shuffled labels 12.1% |
| oracle pass@8 | 38.8% | |
| gap recovered | **27.0%** | |

Script verdict: PARTIAL. Controls clean. But this table has no
critic-free baseline — see §3.

## 2. Fresh corpus `fox_il_v1` (20 files, auto-port, N=16) — `bestofn_fox_il_v1.md`

| decode, 19,146 decision frames | match |
|---|---|
| sampling pass@1 | 3.6% |
| **mode-of-N (critic-free)** | **6.8%** |
| selector Best-of-N | 4.9% |
| oracle pass@16 | 15.0% |

Selector gap recovered 12.2%; **mode-of-N recovered 28%** and beats the
selector.

## 3. Unseen in-distribution (erickfm files 41–60, port 1, N=16) — `bestofn_erickfm_unseen.md`

Added after §2 because §1 never computed the critic-free baseline.

| decode, 48,191 decision frames | match |
|---|---|
| sampling pass@1 | 14.9% |
| **mode-of-N (critic-free)** | **22.9%** |
| selector Best-of-N | 19.0% |
| oracle pass@16 | 43.3% |

Mode-of-N recovers **28%** of the gap here too (same fraction as
off-corpus); the selector recovers 14%.

## Verdict by the pre-registered rule

The rule's NULL clause fires: *"mode-of-N ties the selector"* — it does
better than tie, it **beats the linear selector on both corpora**
(22.9 vs 19.0 in-distribution, 6.8 vs 4.9 fresh). The §1 "PARTIAL" was
measured against sampling only; against the free baseline the bilinear
head is **NULL**: it learned something real (controls clean, top-1 49.8%
vs 11% chance) but nothing a majority vote over the policy's own samples
doesn't already give.

**The positive result is mode-of-N itself.** A critic-free decode —
draw N=16, play the most frequent joint action — recovers ~28% of the
selection headroom on BOTH corpora, with no weights, no critic, no
retrain: pass@1 14.9 → 22.9 in-distribution, 3.6 → 6.8 off-corpus. It
is the first retrain-free decode change since buttons-T=0.5 that moves
a Leg-S number, and it is exactly the direction Bradley's standing
"exhaust v1 retrain-free first" asks for.

## What this changes

1. **Mode-of-N goes to the live rung next, not the selector.** Wire
   `--decode mode-of-n --n 16` into the Agent (one trunk pass, N head
   samples, majority vote; ties → first sample). Cost: N head evals per
   frame, heads are tiny. Then the standard instruments: `loop_report` +
   `coach_report` vs the T=0.5 baseline (8×120 s, CPU dummy), and
   Bradley's live look (the g6 rule: a Leg-S gain is not a play gain
   until a human says so).
2. The MLP selector head (design's PARTIAL branch) is still allowed but
   must now beat **22.9 / 6.8**, not 16.0 / 3.6. Deferred until
   mode-of-N's live result is in — if a majority vote already cashes the
   headroom that lives in the sample distribution, a learned selector
   needs to read something the distribution doesn't, and V's R² of 0.046
   says the linear φ doesn't carry it.
3. V(s) ranks above chance (0.599 vs 0.514) but explains ~5% of return
   variance: the trunk⊕raw linear probe is a weak value function. Keep
   for the CycleSim-rollout branch; do not build on it yet.

## Open instrument note (not resolved here)

`fox_il_v1` sampling pass@1 differs 2× between instruments: Leg S
(`interp_passk`, 2000 sampled decision frames) says 6.9%, `interp_bestofn`
(all 19,146 decision frames) says 3.6%; oracle 24.3 vs 15.0. On erickfm
the two agree (14.5 vs 14.9). Candidates: different decision-frame
filters, or the 2000-frame subsample. Ratios (mode-of-N / sampling,
oracle / sampling) are the comparable numbers until this is pinned.

## Caveats (from the design, still standing)

Master's exact action is the only positive — the metric under-counts
equally-good alternatives, so all "gap recovered" figures are
conservative. Off-distribution states (the bot's own loops) are absent
from every table here; score BOT replays after the live wiring.
