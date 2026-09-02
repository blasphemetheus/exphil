# Critic-selector live gate — VERDICT: FAILED (L9 fourth confirmation)

Pre-registered gate (HANDOFF_2026-09-02 §2.2) applied before exploratory
numbers:

| gate | CRITIC (K=16) | BASE | verdict |
|---|---:|---:|---|
| frozen-input <= 0.20 | 0.48~0.51 [0.34-0.54] | 0.00 | HARD FAIL |
| 7/8 runs to cap | 2/8 (mean 86.5s) | 7/8 (121.1s) | FAIL |

Mechanism: the bilinear S argmax systematically selects the LEAST ACTIVE
of the K candidates — d_up 3.4/min vs BASE 89.7, half of frames frozen,
recovery first-route "none" 66.7% (died-given-none 82.1%), died% 66.7 vs
BASE 46.7. The selector learned "the master's action is usually the
quiet one"; per-decision argmax turns that prior into passivity. The
offline +6.2 (in-dist, first-ever bar clear) did not survive contact.

Standing law update: L9 now has FOUR confirmations (argmax buttons,
mode-of-N x2, score-argmax selection). Mode-seeking decodes die live at
this policy scale — the v1 decode-knob ladder is EXHAUSTED. The
remaining knob family (softmax-sampling over S, temperature on scores)
is the same class; do not retest without a structurally different idea.

Ops notes: fused ar_tiled_stochastic landed for this gate (35 -> 14
ms/decision at K=16); CRITIC arm staleness 1.5-3.0% — the failure is
behavioral, not latency. Mode-of-N's earlier live look ran on the slow
path (~2x budget) — its behavioral verdict stands (same class, same
death) but its staleness context was worse than believed.
