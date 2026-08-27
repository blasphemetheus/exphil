# Can interp see the peak? — instrument audit over the g19 snapshot series

2026-08-20 evening. Question (Bradley): can existing interp work explain
the peak/decay phenomenon, or do we need new tools? Method: point every
relevant existing instrument at the 60 gated snapshots (the first
behavioral ground-truth series we've ever had) and score each against
the gate table. All offline, no training GPU.

## Instrument scorecard

| instrument | result vs gate table | verdict |
|---|---|---|
| Teacher-forced fixture agreement (`eval_policy_on_fixture`, 13 epochs) | FLAT: 99.75-99.94% at offset 5 for ep2 (gate 78), ep9 (82), ep60 (122), ep4 (437) alike | **Blind.** Every epoch retains the mapping; press rates match to 0.1%. |
| CycleSim closed-loop | Cannot score this lineage at ANY decode-lag (0-3): ep4 (437 live) sims chains<=2 | **Blocked** — the LATENCY_ARCHITECTURE queue-embed caveat is real. Fixing it = the one new build that could yield an offline gate predictor. |
| Weight-space per-layer deltas | Cliff ep8->9 (380->82): total delta 8.2, diffuse, GRU-bias-flavored — SMALLER than within-peak ep4->6 (14.1) and same-shaped as the upward flip ep16->17 (8.2) | **No damage event exists.** Behavioral flips are ordinary steps across a razor-thin boundary. Nothing to detect in weight space. |
| Critical-event margins on a common yardstick (NEW: `scripts/margin_trajectory.exs` — all 60 snapshots' signed logits at ep4's own 435 jc + 435 aerial-shine events, one embed, ~8s/policy) | Clean MONOTONE DRIFT, but Spearman vs gate ~= 0 (aerial -0.03, jc +0.20) | **Reveals mechanism direction, does not predict gates.** |

## The two real findings

1. **The decay is 100% closed-loop.** Teacher-forced, every epoch is
   near-perfect; the gate variable is whether the policy's OWN
   trajectory re-enters the cycle. Gate-relevant skill lives in the
   feedback loop, so ONLY closed-loop instruments (dolphin gate, or a
   repaired CycleSim) can rank checkpoints. This kills the cheap-proxy
   hopes for: loss (known), per-source loss decomposition (skipped —
   necessarily flat for the same reason), fixture margins, agreement.
2. **A monotone seesaw under the flat surface**: on the early-peak
   state distribution, jc X-margins FATTEN all run long (p10 2.2 -> 8.0
   by ep59) while aerial-shine B-margins THIN and flip (p10 +1.8 ->
   -4.6; flip fraction 0 -> 0.6-0.8 by ep50+). The optimizer
   continuously trades the B-press boundary near jumpsquat/aerial
   states for X sharpness. Mid-run gate-82 epochs hold fat margins on
   ep4's states (their breaks happen upstream, in states ep4 never
   visits), and late revisit-peaks chain through their own drifted
   distribution (ep42: gate 414 with p10 -0.29 on ep4 states) — which
   is WHY pointwise margins can't rank checkpoints: each policy has
   its own attractor geometry.

   Suggestive tie-in (not tested): AWBC's weight map damps exactly the
   aerial_jump dwell (0.69) and starves "no shine ahead" (0.15) — the
   state families whose B-margins decay. But g15r2 (NO awbc) also
   decayed, so awbc is at most an amplifier, not the cause.

## Answer to the question

Mostly existing tools — the audit itself was done in an afternoon with
them — but the conclusion they deliver is that NO static instrument
can rank checkpoints, so the behavioral gate-sweep (built this
morning, 30s/checkpoint) stays the selection instrument. The one new
build with predictor potential: **CycleSim queue/pipeline-convention
support** (closed-loop, offline, ~2s/policy — would replace the 30-min
dolphin sweep and unlock margin-style break forensics per epoch).
Worth doing when sweep time becomes a bottleneck; not before.

Artifacts: `margins_yardstick_ep4.jsonl`, `scripts/margin_trajectory.exs`,
scratchpad agree_sweep/cs_sweep logs, wdiff.exs pattern.
