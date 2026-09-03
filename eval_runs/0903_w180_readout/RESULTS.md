# w180 readout (09-03 evening) — the context discriminator arm

One knob vs the v1.3/v1.4 line: `--window-size 180` (3 s context vs 1 s).
Same recipe/corpus/AR-head, fresh from the v1 ep10 trunk, seed 904,
2 epochs (~4.3 h wall; note v1.4-long had ~6 epochs total — this arm is
UNDER-TRAINED relative to it by design/time budget).
Policy: `checkpoints/fox_gen_w180_20260903_172430_best_policy.bin`.
Train: val 5.339 after 2 epochs (v1.4-long: 5.324 after its epoch 1,
5.241 after 4 more — NOT directly comparable across window sizes: the
context distribution differs).

Protocol: eval_live_protocol.sh, 4x120 s vs lvl-1 CPU fox, T=0.5,
foreground (see gotcha below), split T05a/T05b. All 4 runs full-length,
staleness healthy.

## 1. Neutral-range scorecard (`neutral_range.md`) — THE GATE, and it MOVED

toward (mean stick-x toward opponent) by |dx| bucket, vs v1.4@60 same
protocol, expert anchor from eval_runs/0901_neutral_range:

| set | 0-20 | 20-40 | 40-70 | frames@100+ (share) |
|---|---:|---:|---:|---:|
| expert | +0.095 | +0.129 | +0.222 | 1.7% |
| **w180** | **+0.052** | **-0.044** | **+0.042** | **0.6%** |
| v1.4@60 | -0.084 | -0.139 | -0.081 | 4.2% |

- **The toward-sign FLIPS positive at 0-20 and 40-70** and moves 2/3 of
  the way to zero at 20-40. %awy (retreat share) drops from 23-33% to
  16-20% (expert: 17-24%).
- Long-range occupancy collapses toward the expert's: w180 spends 0.6%
  of neutral frames beyond dx=100 (expert 1.7%, v1.4 4.2%) — it closes
  distance instead of camping it.
- STILL ABSENT both arms: dash-dance (dash% ~0 vs expert 8-27%) and the
  SH game (jump% ~1% vs expert 13-16%) — the named 2-5 s skills
  (DD JC upsmash) did not appear at 3 s context.
- NEW pathology: grounded laser at point-blank (laserG 29.8% of 0-20
  neutral frames vs expert 0.1%, v1.4 11.3%) — the approach urge found
  a degenerate expression.

## 2. Commitment scorecard (`commitment.md`) — more contact, more punished

w180 23.2 in-threat committals/min vs v1.4 20.4 (expert anchor ~6/min,
0831); P(punished|in-threat) 52.7% vs 44.7%. Composition shift: in-threat
GRABS 98 events vs 39 — the approach converts into grab attempts
(run-up grab was a named absent skill; it now exists in degenerate,
punishable form).

## 3. Loop report — buttons REGRESS hard at matched T=0.5

| arm | d_up/min | loops/min | held-action frac |
|---|---:|---:|---:|
| w180 T=0.5 | 415-433 | 2.5-3.0 | 0.28-0.41 |
| v1.4 T=0.5 | ~101 | 0.49 | — |
| v1.4 T=1.0 | 410 | 0.16 | — |

w180 at HALF temperature sprays buttons like v1.4 at full temperature.
GRAB_WAIT>GRAB_PUMMEL is the dominant cycle again (15 episodes).
Most plausible read: 2 epochs vs ~6 — the buttons head is under-trained
(w180 val 5.339 ≈ v1.4's epoch-1 5.324); the noise may wash out with
epochs. The confound is real: sharpness and context are on different
training-time schedules in this comparison.

## VERDICT (instrument side; Bradley's eye still owed)

**Context is a live lever for the neutral layer.** At 0.4x the epochs,
3 s of context moved exactly the features the diagnosis named (approach
bias, range occupancy) while leaving loss ~flat — this is not a
sharpness effect, it's a different-behavior effect. The
contiguous-BPTT build (unbounded carried state) is CONFIRMED as the
right next plank before it's finished.

Counterweights: buttons noisier at matched T (likely epoch-starved,
unresolved), close-range laser degeneracy, commitment quality slightly
worse. The 2-5 s skills need more than 3 s context or more than context.

Owed: Bradley live look (w180 vs v1.4 at T=0.5, the launcher-port
command in HANDOFF_2026-09-03 §6); optionally +2 more w180 epochs to
de-confound sharpness.

## Operational gotcha (recorded)

`dolphin-emu` (exi-ai build) SIGSEGVs at boot inside a
`systemd-run --user` unit (3/3 crashes in exit-handler
CEXISlippi::writeToFileAsync teardown; the bridge then times out with
enet_disconnected) — but runs fine from a login shell. Eval protocols
must run FOREGROUND (split into <10-min halves for the Bash limit);
training units are unaffected (no Dolphin). Cause not diagnosed (env
delta suspected; DISPLAY/WAYLAND are present in the user manager, so
it isn't that alone).
