# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 55253 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 4.107 | 1.998 | 1.399 | 6.835 | **0.670** |
| given action-state id | | | | | **0.829** |
| given action-state + situation labels | | | | | **0.904** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 3.39% | P(stick up) | 2.36% | 1.43× |
| P(B pressed | stick up) | 18.99% | P(B pressed) | 13.24% | 1.43× |
| P(stick up | B pressed, offstage) | 24.26% | P(stick up | offstage) | 12.59% | 1.93× |
| P(B pressed | stick up, offstage) | 33.27% | P(B pressed | offstage) | 17.27% | 1.93× |
| P(stick side | L/R pressed) | 52.25% | P(stick side) | 46.98% | 1.11× |
| P(stick side | L/R pressed, offstage) | 62.16% | P(stick side | offstage) | 60.37% | 1.03× |
| P(A pressed | stick down) | 15.21% | P(A pressed) | 19.14% | 0.79× |
| P(stick down | A pressed) | 17.93% | P(stick down) | 22.55% | 0.79× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.145**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:5.96 1:0.57 2:2.27 3:0.57 4:0.28 5:0.00 6:0.00 7:0.00 8:64.40 9:0.00 10:0.00 11:0.28 12:1.42 13:2.27 14:1.56 15:20.43 16:0.00
P(stick_y bucket | offstage):
0:4.14 1:0.64 2:5.51 3:0.69 4:0.20 5:0.15 6:0.00 7:0.00 8:74.01 9:0.00 10:0.00 11:0.98 12:1.10 13:2.84 14:0.76 15:8.99 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
