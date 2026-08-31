# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 52547 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.682 | 1.753 | 0.938 | 7.990 | **0.383** |
| given action-state id | | | | | **0.726** |
| given action-state + situation labels | | | | | **0.908** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 2.99% | P(stick up) | 2.85% | 1.05× |
| P(B pressed | stick up) | 20.89% | P(B pressed) | 19.91% | 1.05× |
| P(stick up | B pressed, offstage) | 3.42% | P(stick up | offstage) | 1.73% | 1.97× |
| P(B pressed | stick up, offstage) | 43.90% | P(B pressed | offstage) | 22.23% | 1.97× |
| P(stick side | L/R pressed) | 40.50% | P(stick side) | 40.80% | 0.99× |
| P(stick side | L/R pressed, offstage) | 45.63% | P(stick side | offstage) | 49.58% | 0.92× |
| P(A pressed | stick down) | 34.00% | P(A pressed) | 31.18% | 1.09× |
| P(stick down | A pressed) | 11.67% | P(stick down) | 10.70% | 1.09× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.050**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:5.70 1:0.28 2:3.61 3:0.76 4:0.19 5:0.19 6:0.00 7:0.00 8:85.38 9:0.00 10:0.00 11:0.09 12:0.38 13:1.90 14:0.57 15:0.95 16:0.00
P(stick_y bucket | offstage):
0:3.82 1:0.21 2:2.51 3:0.59 4:0.19 5:0.25 6:0.00 7:0.00 8:90.31 9:0.00 10:0.00 11:0.11 12:0.27 13:0.84 14:0.32 15:0.57 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
