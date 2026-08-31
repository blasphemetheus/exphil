# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 49199 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.527 | 1.605 | 0.937 | 7.731 | **0.338** |
| given action-state id | | | | | **0.689** |
| given action-state + situation labels | | | | | **0.872** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 2.45% | P(stick up) | 2.02% | 1.21× |
| P(B pressed | stick up) | 24.29% | P(B pressed) | 20.02% | 1.21× |
| P(stick up | B pressed, offstage) | 2.67% | P(stick up | offstage) | 1.48% | 1.80× |
| P(B pressed | stick up, offstage) | 44.30% | P(B pressed | offstage) | 24.57% | 1.80× |
| P(stick side | L/R pressed) | 34.88% | P(stick side) | 34.35% | 1.02× |
| P(stick side | L/R pressed, offstage) | 38.34% | P(stick side | offstage) | 37.92% | 1.01× |
| P(A pressed | stick down) | 31.16% | P(A pressed) | 28.87% | 1.08× |
| P(stick down | A pressed) | 13.17% | P(stick down) | 12.20% | 1.08× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.052**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:8.39 1:0.23 2:2.44 3:0.92 4:0.23 5:0.23 6:0.00 7:0.00 8:84.36 9:0.00 10:0.00 11:0.15 12:0.38 13:0.61 14:0.69 15:1.37 16:0.00
P(stick_y bucket | offstage):
0:5.08 1:0.21 2:2.23 3:0.82 4:0.17 5:0.19 6:0.00 7:0.00 8:89.52 9:0.00 10:0.00 11:0.13 12:0.17 13:0.49 14:0.36 15:0.64 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
