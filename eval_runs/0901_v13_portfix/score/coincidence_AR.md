# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 58320 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.054 | 1.771 | 1.268 | 7.258 | **0.834** |
| given action-state id | | | | | **1.058** |
| given action-state + situation labels | | | | | **1.161** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 4.23% | P(stick up) | 2.23% | 1.90× |
| P(B pressed | stick up) | 30.92% | P(B pressed) | 16.31% | 1.90× |
| P(stick up | B pressed, offstage) | 16.84% | P(stick up | offstage) | 9.04% | 1.86× |
| P(B pressed | stick up, offstage) | 44.13% | P(B pressed | offstage) | 23.68% | 1.86× |
| P(stick side | L/R pressed) | 58.54% | P(stick side) | 41.04% | 1.43× |
| P(stick side | L/R pressed, offstage) | 65.61% | P(stick side | offstage) | 51.60% | 1.27× |
| P(A pressed | stick down) | 21.11% | P(A pressed) | 28.43% | 0.74× |
| P(stick down | A pressed) | 14.15% | P(stick down) | 19.06% | 0.74× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.139**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:8.21 1:0.53 2:2.56 3:0.32 4:1.60 5:0.85 6:0.00 7:0.00 8:64.29 9:0.00 10:0.00 11:2.35 12:2.45 13:2.56 14:0.43 15:13.86 16:0.00
P(stick_y bucket | offstage):
0:4.09 1:0.50 2:3.05 3:0.56 4:0.71 5:0.61 6:0.00 7:0.00 8:76.98 9:0.00 10:0.00 11:2.04 12:2.42 13:3.08 14:0.18 15:5.78 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
