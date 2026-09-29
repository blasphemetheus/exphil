# Engagement + jab-discipline scan (2026-09-29)

Two human impressions from the 09-27 demo (Bradley + a friend on Falco,
Mario, G&W, Marth, Roy, Falcon), measured: **"the Mamba is a camper"** and
**"multi-jab comes out more than in expert play"**. Script:
`scripts/engagement_scan.exs`; rows in `eval_runs/0929_engagement/*.jsonl`.
Subject = port 1 (the bot) for demo sessions; the Fox port for the corpus.
Medians over games; small n for the bots — read as direction, not precision.

| feature | Mamba ep2 (n=3) | GRU PPO (n=14) | GRU IL (n=2) | expert Fox (n=25) p10 / p50 / p90 |
| --- | --- | --- | --- | --- |
| median distance to opponent | **27.6** | 25.2 | 29.9 | 18.1 / 21.0 / 25.9 |
| share of frames > 60 units apart | 0.16 | 0.149 | 0.195 | 0.078 / 0.125 / 0.168 |
| approaches / min (dash closing while free) | 28.3 | **22.5** | 30.7 | – / 31.4 / – |
| initiative share (own approaches / all) | 0.585 | **0.506** | 0.576 | – / 0.623 / – |
| lasers / min | **5.31** | 3.52 | 2.24 | 0.0 / 2.12 / 5.08 |
| frames to first hit per stock (median) | 300 | 290 | 340 | – / 272 / – |
| jab1 / min | 0.59 | 0.63 | 2.66 | 0.42 / 1.04 / 1.96 |
| jab2 / min | 0.39 | 0.0 | 1.73 | – / 0.0 / – |
| jab2 per jab1 | **1.0** | 1.0* | 0.375 | 0.0 / 0.0 / 0.5 |
| **A re-presses during jab1, per jab1** | **1.0** | **1.0** | **1.5** | – / **0.0** / – |

\* median of a per-game ratio over few jabs; the per-minute row is the
honest one for the PPO head.

## Camping — the impression is real, and mild

The Mamba stands farther than the expert 90th percentile (27.6 vs 25.9) and
lasers more than the expert 90th percentile (5.3 vs 5.1/min), while its
approach rate and initiative are inside the expert band. So it is not a
wall-camper; it is a Fox that lasers a lot from slightly too far. The GRU
PPO head is the opposite failure: normal range, **fewest approaches and
lowest initiative of the four** — PPO made it wait. The imitation GRU
approaches the most but from the farthest.

**Evals did not capture this.** The fingerprint that gates PPO has no range,
approach, laser or initiative feature. These five belong in it, with the
expert p10–p90 band as the pass region, before the next PPO promotion call.

## Multi-jab — the mechanism is spurious A press-edges, not "more jabs"

The bots jab *less* often than experts (0.6/min vs 1.0). What differs is
what happens inside a jab: experts almost never press A again during jab 1
(median 0 re-presses per jab1; jab2/jab1 median 0). **Every bot re-presses
A during jab 1 about once per jab (1.0, 1.0, 1.5)**, and the game turns
that into jab 2. So the impression "multi-jab" is right, and the cause is
the button channel flickering — an A press-edge the expert never made.

That points at the training/decoding contract, not capacity:
- labels are per-frame button STATES; a held A and a re-pressed A look the
  same to the loss, so the model has never been told "one press";
- buttons are sampled per frame at T=1.0 (the standing decode, because
  argmax collapses); independent per-frame draws produce toggles, and each
  toggle is a press edge.

## How to address it (if the press-edge account holds)

1. **Edge-aware target (training-side, the real fix).** Add a per-button
   "pressed this frame" target (state_t ∧ ¬state_{t−1}) beside the state
   target and train them jointly; at decode a button goes down only if the
   state head says down AND (it was down last frame OR the edge head says
   press). The controller interface is unchanged; the model learns one
   press per jab because the edge target says so. Cost: one extra 8-way
   binary head, a few lines in the loss, and a parity test that the decoded
   state stream is unchanged on held buttons.
2. **Temporal coherence in the button head (cheaper, partial).** The AR
   head already has a prev-buttons variant; up-weight the loss on frames
   where the expert's button state changes (transition frames) so toggles
   cost more than holds. No interface change.
3. **Diagnostic only, not a fix:** re-run the scan with the button
   temperature at 0.5 for A. If re-presses fall to the expert band the
   flicker account is confirmed; the fix still goes into training
   (no bespoke decode rules — feedback 08-29).

The measurement to repeat after any of these: `a_repress_per_jab1` and
`jab2_per_jab1` on 10+ live games, target = expert p90 (0 / 0.5).
