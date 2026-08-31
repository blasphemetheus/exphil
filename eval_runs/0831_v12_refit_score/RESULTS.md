# v1.2 refit — VERDICT: PASSED both gates (2026-08-31 17:05)

Chain: `scripts/v12_refit_chain.sh` (capture on v1.1-AR's frozen trunk →
both-head fit → probe → score). Read the probe first per the chain rule.

## Gate 1 — the wire (probe, `eval_runs/0831_v12_refit_probe/RESULTS.md`)

| checkpoint | L_cond | R_state |
|---|---:|---:|
| **v1.2-ARrefit** | **2.39** | 1.09 |
| v1.2-INDrefit (control) | — | 1.13 |
| v1.1-AR (source) | 1.14 | 1.05 |

**RESTORED** (8a's frozen-trunk reference: 2.67). The frozen-trunk fit
re-forced attribution into the tf-wire, on the better trunk.

## Gate 2 — live recovery (CPU rung, 8×120 s)

| | expert | **v1.2-ARrefit** | v1.2-INDrefit | v1.1-AR (ref) |
|---|---:|---:|---:|---:|
| recovery died % | 12.7 | **28.6** | 57.4 | 28.1 |
| back % | 79.1 | **69.4** | 42.6 | 71.9 |
| first-route airdodge % | 4.8 | **18.4** | 36.2 | 37.5 |
| deaths / game | 3.18* | **1.50** | 3.25 | 1.13 |
| to cap | — | 7/8 | 4/8 | 7/8 |

v1.1-AR's gains KEPT (died% same class, no collapse) and the airdodge
panic route HALVED — the F1 pathology moving in the wire's direction.
The INDrefit control regressing badly isolates the effect to the head.
Oddity noted, not resolved: ARrefit "none" first-route 55.1% (high) with
good outcomes — possibly drift-back-without-special; TBD if it matters.

## Consequences

- **v1.2-ARrefit (`checkpoints/fox_gen_v1.2_ARrefit_policy.bin`) is the
  deployment candidate.** Bradley's live look is the remaining gate
  (g6), and it also gates the `--head` default flip.
- The freeze-then-refit sequence is now the PROVEN recipe for the AR
  line: unfreeze for polish, refit for structure. The open v2 question
  (unfreeze without atrophy) stands.
- Follow-ups launched 17:10: critic ladder rerun on ARrefit (does
  selector-over-mode widen with a real wire?) + G3b dynamics spike.
