# Joint-head audit — within-frame dependency the independent head cannot represent

24 expert files, port 1, 176239 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.038 | 1.781 | 1.273 | 7.261 | **0.832** |
| given action-state id | | | | | **0.955** |
| given action-state + situation labels | | | | | **1.048** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 3.36% | P(stick up) | 2.11% | 1.60× |
| P(B pressed | stick up) | 28.05% | P(B pressed) | 17.57% | 1.60× |
| P(stick up | B pressed, offstage) | 10.27% | P(stick up | offstage) | 5.44% | 1.89× |
| P(B pressed | stick up, offstage) | 43.74% | P(B pressed | offstage) | 23.18% | 1.89× |
| P(stick side | L/R pressed) | 58.79% | P(stick side) | 41.14% | 1.43× |
| P(stick side | L/R pressed, offstage) | 62.56% | P(stick side | offstage) | 48.37% | 1.29× |
| P(A pressed | stick down) | 21.40% | P(A pressed) | 27.37% | 0.78× |
| P(stick down | A pressed) | 15.71% | P(stick down) | 20.09% | 0.78× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.125**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:8.28 1:0.37 2:4.80 3:0.91 4:0.33 5:0.95 6:0.00 7:0.00 8:72.13 9:0.00 10:0.00 11:0.87 12:1.08 13:5.59 14:0.50 15:4.18 16:0.00
P(stick_y bucket | offstage):
0:3.94 1:0.21 2:3.34 3:0.50 4:0.19 5:0.45 6:0.00 7:0.00 8:84.58 9:0.00 10:0.00 11:0.47 12:0.87 13:3.22 14:0.33 15:1.90 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
