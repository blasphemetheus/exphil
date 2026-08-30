# Live look — AWBC arms vs Bradley, 2026-08-29 evening (FD, delay 0, T=0.5 / buttons 0.5)

Blind? No — Bradley knew which arm he was playing. Same evening, same
human, same decode; replays under eval_runs/0829_livelook_awbc_B*/.
Score with loop_report + coach_report after all arms are played.

## B2 (AWBC, standard reward) — played first

Bradley, verbatim-ish: "definitely felt better than v1. Still does
grab-to-pummel way too long, does standing lasers, and sometimes goes
sideways off the stage. On the whole it feels harder to hit. Still does
multi-jab → multi-jab a lot, but not as much as it used to. Plays the
scraps a little better than the previous version."

Note: against the CPU dummy B2 measured identical to B1/B3 (0828_awbc_arms
RESULTS, NULL). A human-felt difference that the CPU rung cannot see is
exactly the g6 pattern in reverse — record it, then test it: B1 and B3
back-to-back tonight, and the replays scored on the same instruments.

## B1 (plain 3-epoch continuation) — played second

Bradley: "B1 is better than v1. I think B1 is probably better than B2,
but they play different. B2 spams options a little more — like a
spot-dodger. They play different enough that it is hard to say which is
better."

Read so far: both 3-epoch continuations beat ep10 by feel; B2 vs B1 is a
style difference (B2 more option-spam / spot dodge), not a clear
ordering. The "AWBC helps" reading from the first game is NOT supported
once B1 is in hand — the shared gain is the three extra epochs. B3
decides whether B2's style shift is the outcome signal or the weight
distribution.
## B3 (AWBC, shuffled weights — placebo control) — played third

Bradley: "B3 also plays a little different. It has more of the
pathologies of spamming things in place, but it also has its own
character. Kind of interesting."

Read: three continuations, three styles, no ordering a human could
call. B3 (no outcome information, same weight distribution as B2) has
its own character too — so "B2 feels different" is consistent with
non-uniform weights or just seed-level variation between 3-epoch runs,
not with the outcome signal. Matches the CPU verdict (NULL) on the
causal question; adds one thing the CPU could not see: all three
continuations beat ep10 by feel. Score the replays next.

## Scored (loop_report --bot-port 1, coach_report --char fox; CSS stubs <150 KB excluded)

| arm | games | d_up/min [range] | loops/min | max loop reps | held-action | deaths/game | conversions | dropped/game | neutral loss/game |
|---|---|---|---|---|---|---|---|---|---|
| B1 plain | 7 | 92.5 [78–99] | **0.22** | 1.0 | 0.24 | 3.29 | 5/34 (15%) | 0.71 | 5.3 |
| B2 AWBC | 10 | 109.8 [88–129] | 0.94 | 3.2 | 0.25 | 3.80 | 14/74 (19%) | 1.30 | 4.9 |
| B3 shuffled | 5 | 102.1 [96–114] | 0.81 | 2.2 | 0.21 | 2.80 | 6/45 (13%) | 1.20 | 5.4 |

Read against the impressions:
- B1's "probably better" has a number: loops/min 0.22 vs 0.94 / 0.81 —
  4× fewer repeated cycles than B2, ranges [0–0.9] vs [0–2.1] (touching,
  n=7/10). B1 also has the lowest d_up rate (92.5, disjoint from B2's
  109.8 — but this is the 1.19× band, unresolved by the 2× law).
- B2's "spams options / spot-dodger": loops/min 0.94, max loop reps 3.2,
  the most conversions attempted (74 openings in 10 games) and the most
  dropped (1.3/game) — busier, sloppier. Deaths highest (3.80).
- B3's "in-place spam": loops/min 0.81, in between; fewest deaths (2.80,
  n=5).
- Nothing here separates ≥2× with disjoint ranges except B1's loops/min
  against B2 (4.3×, ranges touch at 0). Consistent with "three styles,
  B1 the cleanest," and with the CPU NULL: the outcome weighting did not
  produce a better player; the plain continuation is the tidiest of the
  three.
- All three vs ep10's last human session (T=0.5, 08-28: d_up 101, deaths
  3.57, dropped 2.6): same d_up band, same deaths; dropped conversions
  are LOWER in all three continuations (0.7–1.3 vs 2.6). That is the one
  number that moved with the extra epochs, and it matches "harder to hit
  / plays the scraps better" only loosely. Worth a B1 vs ep10 blind pair
  if it matters.
