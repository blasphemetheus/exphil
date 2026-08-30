# Joint-head audit — within-frame dependency the independent head cannot represent

1500 expert files, port 1, 16178232 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 1.912 | 2.597 | 1.998 | 5.651 | **0.856** |
| given action-state id | | | | | **0.841** |
| given action-state + situation labels | | | | | **0.858** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 23.68% | P(stick up) | 7.37% | 3.21× |
| P(B pressed | stick up) | 14.81% | P(B pressed) | 4.61% | 3.21× |
| P(stick up | B pressed, offstage) | 41.65% | P(stick up | offstage) | 14.57% | 2.86× |
| P(B pressed | stick up, offstage) | 37.47% | P(B pressed | offstage) | 13.11% | 2.86× |
| P(stick side | L/R pressed) | 59.62% | P(stick side) | 50.40% | 1.18× |
| P(stick side | L/R pressed, offstage) | 69.37% | P(stick side | offstage) | 58.18% | 1.19× |
| P(A pressed | stick down) | 3.93% | P(A pressed) | 5.04% | 0.78× |
| P(stick down | A pressed) | 12.01% | P(stick down) | 15.41% | 0.78× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.359**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:6.65 1:1.54 2:3.22 3:2.03 4:1.06 5:1.35 6:0.50 7:0.00 8:31.62 9:0.00 10:0.99 11:5.67 12:3.71 13:6.33 14:15.24 15:6.44 16:13.63
P(stick_y bucket | offstage):
0:3.63 1:1.14 2:2.31 3:1.54 4:0.97 5:1.36 6:0.62 7:0.00 8:67.41 9:0.00 10:0.80 11:3.56 12:2.10 13:3.02 14:5.78 15:1.97 16:3.81

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
