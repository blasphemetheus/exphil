# Plan (c) V-rollout selector — offline RESULTS (v3, discriminative g)

Policy fox_gen_v1.2_ARrefit_policy.bin, dynamics dynamics_fox_v11AR.bin
(1-step R^2 0.995, cos@10 0.824).
Scorer g = 256x256 MLP on trunk(imagined window: 50 real + 10
predicted embeds under the held candidate), trained discriminatively
(binary CE, master=1 vs 16 policy samples=0) on 6000 train decision rows
from 25 replays. Eval: 3000 decision rows from
8 held-out replays, K=16 coherent candidates at T=0.5.

| metric | value |
|---|---:|
| sampling pass@1 | 0.7% |
| **rollout-g selector pass@1** | **0.9%** |
| selector, shuffled-STATE control | 1.0% |
| oracle pass@16 | 10.2% |
| master top-1 among K+1 (chance 5.9%) | 9.8% |
| mean g: master / samples | 0.021 / 0.018 |

Gap recovered: 2.2% of (oracle - sampling).

Design history (git): v1 linear V-on-embeds ANTI-correlated (0.406); v1b MLP
V-on-embeds chance (0.503); v2 RTG-trained V on imagined trunk feats works
as a value fn (rank 0.554) but cannot separate candidates (selector 0.5 vs
sampling 0.7) -> v3 trains the scorer discriminatively on the same features.

Reference (same corpus, direct bilinear critic on ARrefit,
eval_runs/0831_critic_refit): sampling 1.3 / mode-of-N 7.2 / selector 10.8 /
oracle 13.9 (different row rule — compare within-table margins only).

Caveats: candidate held constant through the rollout; imagined embeds are
approximate trunk inputs; match-rate is a mode-seeking proxy — the live
gate (frozen-input <= 0.20, 7/8 cap, F1 airdodge, F2 commitment, F3
approach_delta) stays the real judge before any decode change.

## CAMPAIGN VERDICT (v1 → v3 + action-sensitivity, 09-01)

**Plan (c) offline: NEGATIVE at this rung — and the diagnosis is
informative.** The full chain:

1. Dynamics model is NOT the problem: it passed G3b (R² 0.996, cos@10
   0.827) AND the action-sensitivity diagnostic (held actions diverge
   trajectories by 15–33% of the rollout's own drift —
   `action_sensitivity.md`; the G3b gate alone could not have shown
   this, worth keeping as a standard check).
2. Value-on-consequences is the problem: an RTG-trained V cannot
   separate candidates (v2: selector 0.5 vs sampling 0.7), and even a
   DISCRIMINATIVELY trained g on the same imagined-window features barely
   beats sampling with a shuffled-state control at the same level
   (v3: 0.9 vs 0.7 vs 1.0 shuffled; master-top1 9.8% vs 5.9% chance).
3. Read together with the critic ladder's own control (0831_critic_refit:
   shuffled 11.5 vs selector 16.5 — "most of the lift is generic
   action-frequency preference"): the rollout scorer physically CANNOT
   use action identity except through imagined consequences, and once
   that shortcut is removed there is ~no state-conditional
   master-matching signal left at a 10-frame horizon. The direct
   critic's +3.6 margin and the rollout's +0.2 agree: the offline
   "match the master at decision frames" objective is mostly not
   selectable from short-horizon information.

**Implication:** the selection program's offline rung is exhausted —
master-matching is the wrong objective to keep optimizing. What remains
live for plan (c) is a different use of the same machinery: VETO
selection (score candidates for catastrophe — offstage drift, panic
routes — where 10-frame consequences ARE informative, cf. F1) rather
than "pick the master's exact action." That is a live-gate question.
The stronger lever per F3c stands: the approach data exists in the
corpus and BC loses it — training-side (curation weighting / AWBC on
neutral wins), not decode-side.

Machinery banked for reuse: saved dynamics model (+ `--save` on the
spike), imagined-window-through-trunk features, discriminative g
harness, action-sensitivity diagnostic.
