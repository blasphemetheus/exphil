# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 50393 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.777 | 2.158 | 1.543 | 8.167 | **1.311** |
| given action-state id | | | | | **1.721** |
| given action-state + situation labels | | | | | **1.922** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 5.60% | P(stick up) | 2.55% | 2.20× |
| P(B pressed | stick up) | 39.33% | P(B pressed) | 17.90% | 2.20× |
| P(stick up | B pressed, offstage) | 6.62% | P(stick up | offstage) | 2.50% | 2.65× |
| P(B pressed | stick up, offstage) | 56.64% | P(B pressed | offstage) | 21.41% | 2.65× |
| P(stick side | L/R pressed) | 67.60% | P(stick side) | 56.04% | 1.21× |
| P(stick side | L/R pressed, offstage) | 66.73% | P(stick side | offstage) | 59.18% | 1.13× |
| P(A pressed | stick down) | 29.64% | P(A pressed) | 31.59% | 0.94× |
| P(stick down | A pressed) | 23.39% | P(stick down) | 24.93% | 0.94× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.270**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:27.53 1:0.65 2:17.57 3:1.96 4:0.65 5:1.31 6:0.00 7:0.00 8:42.89 9:0.00 10:0.00 11:0.49 12:0.33 13:3.02 14:0.49 15:3.10 16:0.00
P(stick_y bucket | offstage):
0:10.48 1:0.33 2:12.51 3:2.12 4:0.58 5:1.43 6:0.00 7:0.00 8:69.59 9:0.00 10:0.00 11:0.24 12:0.21 13:1.00 14:0.14 15:1.36 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
