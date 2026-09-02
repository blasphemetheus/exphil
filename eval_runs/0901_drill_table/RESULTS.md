# Hit-confirm drill table — expert conversion cells

7523 conversions from 200 files; window
240 f after hit 1; cells with n >= 25, ranked by
n x (mean_hits - 1) ("how much continuation this cell teaches").

| opener | victim % | zone | n | mean hits | >=3 hits % | mean dmg | stock % |
|---|---|---|---:|---:|---:|---:|---:|
| other | 0-19% | mid | 472 | 3.8 | 75 | 7.6 | 0 |
| other | 0-19% | side | 405 | 3.6 | 65 | 6.4 | 0 |
| dair | 0-19% | mid | 181 | 5.3 | 92 | 13.7 | 1 |
| other | 20-39% | mid | 188 | 3.6 | 71 | 6.8 | 0 |
| special | 0-19% | mid | 168 | 3.8 | 71 | 11.0 | 0 |
| other | 20-39% | side | 155 | 3.7 | 69 | 7.0 | 1 |
| special | 0-19% | side | 147 | 3.5 | 61 | 7.5 | 1 |
| dair | 0-19% | side | 86 | 5.1 | 86 | 11.8 | 1 |
| other | 40-59% | side | 153 | 3.3 | 66 | 7.4 | 1 |
| other | 40-59% | mid | 147 | 3.3 | 67 | 7.5 | 0 |
| other | 60-79% | mid | 141 | 3.4 | 68 | 7.2 | 1 |
| other | 60-79% | side | 125 | 3.6 | 74 | 7.3 | 2 |
| other | 80-99% | side | 146 | 3.1 | 66 | 6.4 | 3 |
| bair | 0-19% | mid | 96 | 4.0 | 79 | 12.5 | 0 |
| other | 80-99% | mid | 115 | 3.4 | 73 | 5.0 | 3 |
| nair | 20-39% | mid | 67 | 4.3 | 82 | 12.0 | 1 |
| nair | 0-19% | mid | 80 | 3.6 | 75 | 9.8 | 0 |
| nair | 40-59% | mid | 82 | 3.5 | 74 | 10.6 | 2 |
| bair | 60-79% | mid | 100 | 3.0 | 63 | 9.9 | 1 |
| bair | 40-59% | mid | 87 | 3.3 | 70 | 8.5 | 0 |

## Throw cells (all, regardless of rank — Drill 1 candidates)

| opener | victim % | zone | n | mean hits | >=3 hits % | mean dmg | stock % |
|---|---|---|---:|---:|---:|---:|---:|
| uthrow | 0-19% | mid | 39 | 3.9 | 87 | 27.0 | 0 |

Drill 1 pre-registration check (DRILL_HITCONFIRM.md): the uthrow rows in
the 0-39% bands are the candidate cells — their mean-hits / >=3-hit /
damage columns are the reference distributions the drill scores against.

## Mining notes (09-01, three detector iterations — all in git)

1. Opener at the hitstun edge is usually already IASA'd → 10-frame
   lookback for the last attack-family action.
2. Thrown victims carry hitstun_frames_left == 0 → hit events must
   include THROWN (239-243) and captured (223-232) states. **F4's depth
   counter shares this gap (throw links uncounted) — its flat-across-
   depth verdict is qualitative-safe but depths are shifted; note for
   the re-run backlog.**
3. Thrower action ids were off by two in pass 1-3 (correct: fthrow 219,
   bthrow 220, uthrow 221, dthrow 222); grab openers resolve FORWARD to
   the throw within 120 f, window anchored at the throw.

## Drill 1 reference (pre-registered cell, now measured)

**uthrow / 0-19% / mid: n=39, mean 3.9 hits, 87% >=3 hits, 27.0 mean
damage** — the highest mean-damage cell in the table. Drill scoring
targets these three numbers.

Drill 2 candidate (data-ranked): **dair / 0-19% / mid: n=181, mean 5.3
hits, 92% >=3, 13.7 dmg** — the platform-fox dair chains (the "unused
artifact" Bradley named); 4.6x the uthrow cell's volume.

Refinement backlog: the :other opener bucket is still the largest
(lasers + unmapped ids) — split it before trusting cross-opener ranks.
