# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 57863 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 4.064 | 1.902 | 1.629 | 6.678 | **0.916** |
| given action-state id | | | | | **1.090** |
| given action-state + situation labels | | | | | **1.177** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 2.48% | P(stick up) | 3.27% | 0.76× |
| P(B pressed | stick up) | 12.78% | P(B pressed) | 16.83% | 0.76× |
| P(stick up | B pressed, offstage) | 6.34% | P(stick up | offstage) | 4.96% | 1.28× |
| P(B pressed | stick up, offstage) | 23.08% | P(B pressed | offstage) | 18.04% | 1.28× |
| P(stick side | L/R pressed) | 60.12% | P(stick side) | 45.74% | 1.31× |
| P(stick side | L/R pressed, offstage) | 77.30% | P(stick side | offstage) | 75.90% | 1.02× |
| P(A pressed | stick down) | 8.89% | P(A pressed) | 13.89% | 0.64× |
| P(stick down | A pressed) | 19.44% | P(stick down) | 30.35% | 0.64× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.083**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:5.16 1:0.23 2:2.82 3:1.41 4:1.64 5:2.11 6:0.00 7:0.00 8:74.18 9:0.00 10:0.00 11:4.23 12:1.88 13:4.93 14:0.00 15:1.41 16:0.00
P(stick_y bucket | offstage):
0:3.90 1:0.51 2:1.61 3:0.76 4:0.97 5:1.52 6:0.00 7:0.00 8:81.66 9:0.00 10:0.00 11:2.41 12:1.69 13:3.05 14:0.08 15:1.82 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
