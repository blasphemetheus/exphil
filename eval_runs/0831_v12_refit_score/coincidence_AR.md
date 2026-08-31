# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 57602 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.677 | 2.008 | 1.568 | 7.978 | **1.276** |
| given action-state id | | | | | **1.641** |
| given action-state + situation labels | | | | | **1.842** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 4.54% | P(stick up) | 2.27% | 2.00× |
| P(B pressed | stick up) | 37.72% | P(B pressed) | 18.85% | 2.00× |
| P(stick up | B pressed, offstage) | 8.19% | P(stick up | offstage) | 3.10% | 2.64× |
| P(B pressed | stick up, offstage) | 64.79% | P(B pressed | offstage) | 24.51% | 2.64× |
| P(stick side | L/R pressed) | 60.36% | P(stick side) | 46.25% | 1.30× |
| P(stick side | L/R pressed, offstage) | 61.50% | P(stick side | offstage) | 48.29% | 1.27× |
| P(A pressed | stick down) | 27.78% | P(A pressed) | 29.03% | 0.96× |
| P(stick down | A pressed) | 25.93% | P(stick down) | 27.10% | 0.96× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.264**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:32.32 1:0.80 2:11.84 3:2.58 4:0.36 5:1.42 6:0.00 7:0.00 8:41.05 9:0.00 10:0.00 11:0.53 12:0.89 13:3.12 14:0.71 15:4.36 16:0.00
P(stick_y bucket | offstage):
0:14.87 1:0.52 2:9.65 3:1.86 4:0.39 5:1.44 6:0.00 7:0.00 8:67.39 9:0.00 10:0.00 11:0.44 12:0.35 13:1.16 14:0.28 15:1.66 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
