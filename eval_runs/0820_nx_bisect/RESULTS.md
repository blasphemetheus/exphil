# nx bug isolation — commit-level probes (2026-08-20)

Direction from Bradley 08-20 morning: use correct gradients; isolate
the nx bug causing the GOTCHA #99 collapses. Range a7497612..f843aa1a
narrowed by inspection: all lib changes arrived in ONE merge
(7066052f) of upstream PRs #1814–#1819; only two touch anything that
runs during training — #1814 (donation marks, exla defn compile path)
and #1815 (multi-tensor impl! dispatch, eager ops). Probe = 20-epoch
g18 recipe, classifier from 0819_nxpin_probe (WILD = one-epoch ratio
>8x or loss <1e-4; gray zone 4-10x → rerun).

## Probes

| probe | nx state | verdict |
|---|---|---|
| baseline (0819_nxpin_probe) | a7497612 | CALM (max 4.2x over 83 epochs) |
| baseline (g15r2/g18a1/g18a2) | f843aa1a | WILD (10.7x / collapse / 153x) |
| **p1814** | a7497612 + cherry-pick 5ad4ec45 (#1814 ONLY) | **WILD (13.8x in 20 epochs)** |
| p1815 | a7497612 + #1815 only | not yet run (control; run if scene analysis is ambiguous) |

## Verdict so far

> **RETRACTED 2026-08-20 ~10:00** — see
> `eval_runs/0820_collapse_forensics/RESULTS.md`. The classifier this
> probe used reads the LAST-BATCH epoch loss, which is a heavy-tail
> lottery on BOTH stacks (calm stack drew a 1.67 batch vs 0.035
> median the same morning). p1814's "WILD 13.8x" is within tail-draw
> odds; single-step replay, batch-loss distributions, and 2-epoch
> cumulative damage are all stack-IDENTICAL. #1814: unproven, likely
> innocent. The real phenomenon is unstable behavioral peaks +
> loss/behavior anti-correlation (stopping-point luck), stack-
> independent.

Original (pre-retraction) text: PR #1814 alone reproduces the wild
dynamics on the calm base. Branch kept at `bisect/a74-plus-1814`
(nx repo) for replays.

Mechanism hypothesis (untested, being probed by the collapse-forensics
run): #1814's exla/defn.ex change rebuilds the compile-cache OUTPUT
TEMPLATE via `Nx.Defn.Composite.traverse(&%{&1 | donatable?: false})`.
That template decodes EXLA result buffers back into the train-step's
output container (new params + optimizer state + metrics). If the
traverse alters traversal/rebuild order even subtly, same-shaped f32
tensors get MISASSIGNED on decode — scrambled-but-valid updates:
mostly destructive (wild epochs, dead exports), occasionally a jackpot
(362.5/419.4). Matches the phenotype exactly. The forensics scene
replay can test it directly: same batch + params through train_step on
both stacks, diff the OUTPUT params tensor-by-tensor — a permutation
signature is unmistakable.

## Status / next

- Collapse-forensics run live (eval_runs/0820_collapse_forensics/):
  warm start from g15r2-ep50 on f843aa1a, --collapse-forensics armed.
- After mechanism confirmation: report upstream (nx #1814 follow-up)
  with the minimal repro; pin exphil training to a7497612 (or
  integration-with-1814-reverted) until fixed.
- p1815 control probe only if needed.
