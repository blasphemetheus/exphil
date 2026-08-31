# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 15573 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 0.234 | 1.520 | 0.361 | 2.035 | **0.081** |
| given action-state id | | | | | **0.086** |
| given action-state + situation labels | | | | | **0.091** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 0.00% | P(stick up) | 0.21% | 0.00× |
| P(B pressed | stick up) | 0.00% | P(B pressed) | 0.94% | 0.00× |
| P(stick up | B pressed, offstage) | 0.00% | P(stick up | offstage) | 0.22% | 0.00× |
| P(B pressed | stick up, offstage) | 0.00% | P(B pressed | offstage) | 1.25% | 0.00× |
| P(stick side | L/R pressed) | 30.00% | P(stick side) | 35.80% | 0.84× |
| P(stick side | L/R pressed, offstage) | 0.00% | P(stick side | offstage) | 34.57% | 0.00× |
| P(A pressed | stick down) | 0.45% | P(A pressed) | 0.18% | 2.52× |
| P(stick down | A pressed) | 7.14% | P(stick down) | 2.84% | 2.52× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.041**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:0.00 1:0.00 2:0.00 3:0.00 4:4.35 5:0.00 6:0.00 7:0.00 8:95.65 9:0.00 10:0.00 11:0.00 12:0.00 13:0.00 14:0.00 15:0.00 16:0.00
P(stick_y bucket | offstage):
0:1.22 1:0.11 2:1.06 3:0.35 4:0.22 5:0.27 6:0.00 7:0.00 8:96.56 9:0.00 10:0.00 11:0.00 12:0.00 13:0.00 14:0.03 15:0.19 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
