# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 58258 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 5.027 | 1.350 | 1.163 | 7.090 | **0.452** |
| given action-state id | | | | | **0.617** |
| given action-state + situation labels | | | | | **0.697** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 2.03% | P(stick up) | 2.03% | 1.00× |
| P(B pressed | stick up) | 16.39% | P(B pressed) | 16.42% | 1.00× |
| P(stick up | B pressed, offstage) | 5.57% | P(stick up | offstage) | 2.59% | 2.15× |
| P(B pressed | stick up, offstage) | 36.90% | P(B pressed | offstage) | 17.19% | 2.15× |
| P(stick side | L/R pressed) | 28.42% | P(stick side) | 23.93% | 1.19× |
| P(stick side | L/R pressed, offstage) | 43.03% | P(stick side | offstage) | 42.89% | 1.00× |
| P(A pressed | stick down) | 23.36% | P(A pressed) | 30.06% | 0.78× |
| P(stick down | A pressed) | 14.39% | P(stick down) | 18.51% | 0.78× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.055**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:4.31 1:0.18 2:4.85 3:0.54 4:0.36 5:0.18 6:0.00 7:0.00 8:82.59 9:0.00 10:0.00 11:0.54 12:0.90 13:4.67 14:0.36 15:0.54 16:0.00
P(stick_y bucket | offstage):
0:3.09 1:0.31 2:4.01 3:0.34 4:0.22 5:0.09 6:0.00 7:0.00 8:87.57 9:0.00 10:0.00 11:0.59 12:1.20 13:1.79 14:0.25 15:0.56 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
