# Joint-head audit — within-frame dependency the independent head cannot represent

8 expert files, port 1, 17307 frames. Components audited: buttons (7-bit
set), main-stick x (17 buckets), main-stick y (17 buckets). Bits per frame.

## Total correlation of (buttons, main_x, main_y)

| conditioning | H(buttons) | H(main_x) | H(main_y) | H(joint) | **TC = ΣH − H(joint)** |
|---|---:|---:|---:|---:|---:|
| none (upper bound) | 0.163 | 1.468 | 0.493 | 1.970 | **0.154** |
| given action-state id | | | | | **0.121** |
| given action-state + situation labels | | | | | **0.117** |

TC is the KL divergence between the true joint and the product of its marginals — the
minimum extra cross-entropy (bits/frame) an independent-heads model pays even with a perfect
trunk, IF the trunk carried no more about the frame than the conditioning variables. The
true trunk knows more than an action id + labels, so the last row is an estimate of the
floor of the loss, not a bound on it; the pair statistics below are the part that no state
information can remove because they are about the SAME frame's inputs.

## Pairs that matter (expert)

| conditional | value | marginal | value | ratio |
|---|---:|---|---:|---:|
| P(stick up | B pressed) | 0.00% | P(stick up) | 0.08% | 0.00× |
| P(B pressed | stick up) | 0.00% | P(B pressed) | 0.96% | 0.00× |
| P(stick up | B pressed, offstage) | 0.00% | P(stick up | offstage) | 0.06% | 0.00× |
| P(B pressed | stick up, offstage) | 0.00% | P(B pressed | offstage) | 1.12% | 0.00× |
| P(stick side | L/R pressed) | 25.00% | P(stick side) | 37.38% | 0.67× |
| P(stick side | L/R pressed, offstage) | 0.00% | P(stick side | offstage) | 47.75% | 0.00× |
| P(A pressed | stick down) | 0.08% | P(A pressed) | 0.11% | 0.73× |
| P(stick down | A pressed) | 5.26% | P(stick down) | 7.22% | 0.73× |

## Up-B: stick-y when B is pressed offstage vs stick-y offstage generally

TV distance between P(stick_y | B, offstage) and P(stick_y | offstage): **0.031**.
P(stick_y bucket | B, offstage), buckets 0..16 (0 = full down, 16 = full up):
0:2.50 1:0.00 2:0.00 3:0.00 4:0.00 5:0.00 6:0.00 7:0.00 8:97.50 9:0.00 10:0.00 11:0.00 12:0.00 13:0.00 14:0.00 15:0.00 16:0.00
P(stick_y bucket | offstage):
0:1.54 1:0.62 2:1.57 3:0.14 4:0.17 5:0.28 6:0.00 7:0.00 8:95.35 9:0.00 10:0.00 11:0.11 12:0.17 13:0.06 14:0.00 15:0.00 16:0.00

## Read

- A ratio far from 1× in the pair table is a dependency between two heads on the SAME
  frame. An independent head samples them separately: e.g. if P(up | B) = 60% but
  P(up) = 10%, an independent model that has learned both marginals produces
  "B with stick up" 10% of the B presses — one up-B in ten B presses; the expert does six.
- The state can only rescue this if the trunk state predicts the press itself sharply
  (then P(B|s) ≈ 1 and P(up|s) ≈ 1 on the same frame). B3 measured the buttons head at
  2.5–4 bits at deploy: it does not.
