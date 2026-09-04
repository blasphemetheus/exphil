# w180b readout (09-04 ~03:20, autonomous overnight) — window-180 at matched compute

w180 resumed +4 epochs (6 total ≈ v1.4-long's budget), warm cache,
~1.5 h/epoch. Policy: `checkpoints/fox_gen_w180b_20260904_020807_best_policy.bin`.
Protocol: same as 0903 readouts — 4x120 s vs lvl-1 CPU fox, T=0.5,
foreground halves (T05a/T05b). Two runs ended early by game end
(4405/5391 frames — SD-outs; see caveats).

## 1. Loss: NEW ALL-TIME BEST fox_gen val, monotone, no plateau

5.339 (e2) -> 5.2886 -> 5.2552 -> 5.2291 -> **5.2082** (e6).
Beats v1.4-long's 5.2409 at comparable compute with 3x context.
Underfit story holds at window 180; context costs nothing on loss and
now buys more than window-60 did.

## 2. Buttons: the d_up spray does NOT wash out — it is a WINDOW property

| arm | d_up/min | loops/min |
|---|---:|---:|
| w180b (6 ep) | 429-450 | ~1.0-1.5 |
| w180 (2 ep) | 415-433 | 2.5-3.0 |
| v1.4@60 (T=0.5) | ~101 | 0.49 |

Epochs fixed loops but not the spray: at matched T=0.5 the window-180
line presses d_up 4x more than window-60, unchanged from epoch 2 to
epoch 6. The 0903 "epoch-starved" hypothesis is REFUTED — this is
structural to window 180 under the current recipe (last-frame-only
supervision over a 3x window dilutes per-button pressure? d_up is a
near-free no-op to the loss). Candidate fixes live in the BPTT plank
(per-timestep supervision changes exactly this) — measure there before
inventing a knob here (no bespoke decode rules).

## 3. Neutral: the approach gradient STRENGTHENS toward expert shape

toward by |dx| (expert anchor / w180b / w180e2):

| range | expert | w180b | w180e2 |
|---|---:|---:|---:|
| 40-70 | +0.222 | **+0.106** | +0.042 |
| 70-100 | +0.435 | **+0.393** | +0.166 |

At 70-100 the arm is near-expert approach commitment (n=892 frames);
retreat share at 40-100 is at-or-below expert (14.5%/4.8%). The
rising-toward-with-distance SHAPE — absent in every window-60 arm —
is now visible. Persisting absences: dash-dance ~0% (expert 27% at
40-70), SH/jump game ~1% (expert 14%); persisting degeneracy:
point-blank laserG 31% at 0-20. Close range (0-20) toward slipped to
-0.015 (was +0.052) — mixed at the closest band.

## 4. Commitment: epochs bought quality back

in-threat committals 18.0/min (e2: 23.2; v1.4: 20.4);
P(punished|in-threat) 43.8% (e2: 52.7; v1.4: 44.7). The approach no
longer costs commitment quality — w180b matches v1.4's punish profile
while approaching.

## VERDICT

Context (3 s) at matched compute: best-ever loss, near-expert approach
gradient at mid-long range, commitment quality recovered — and one
clean structural regression (d_up spray) that per-timestep supervision
(BPTT plank D) is the principled fix for. The context lever is
CONFIRMED at strength; proceed with the BPTT build.

Caveats: n=4 runs/day/opponent as always; 2 early game-ends suggest
elevated SD rate (self/min 3.9-6.3 in the shine scorer) — worth a
column in the next readout; Bradley live look owed on w180b.
