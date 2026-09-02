# Joint-head audit — within-frame dependency the independent head cannot represent

24 expert files, port 1, 176438 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.017 | 1.307 | 1.102 | 7.044 | **0.383** |
| given action-state id | | | | | **0.483** |
| given action-state + situation labels | | | | | **0.562** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 2.83% | P(stick up) | 2.40% | 1.18× |
| P(B pressed | stick up) | 18.24% | P(B pressed) | 15.47% | 1.18× |
| P(stick up | B pressed, offstage) | 9.21% | P(stick up | offstage) | 4.37% | 2.11× |
| P(B pressed | stick up, offstage) | 36.45% | P(B pressed | offstage) | 17.30% | 2.11× |
| P(stick side | L/R pressed) | 27.54% | P(stick side) | 23.15% | 1.19× |
| P(stick side | L/R pressed, offstage) | 32.94% | P(stick side | offstage) | 33.02% | 1.00× |
| P(A pressed | stick down) | 23.12% | P(A pressed) | 31.74% | 0.73× |
| P(stick down | A pressed) | 11.60% | P(stick down) | 15.92% | 0.73× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.106**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:8.36 1:0.61 2:3.94 3:0.55 4:0.30 5:0.48 6:0.00 7:0.00 8:75.23 9:0.00 10:0.00 11:0.61 12:0.73 13:3.63 14:0.61 15:4.91 16:0.06
P(stick_y bucket | offstage):
0:5.14 1:0.31 2:2.37 3:0.41 4:0.20 5:0.26 6:0.00 7:0.00 8:85.76 9:0.00 10:0.00 11:0.38 12:0.80 13:2.11 14:0.35 15:1.91 16:0.01

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
