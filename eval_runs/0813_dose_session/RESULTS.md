# Dose-response taste test — 2026-08-12 evening (HANDOFF_2026-08-12 item 0)

Sealed order: g1=convC2, g2=convC4, g3=convC3. Bradley port 2, one game each,
temp 0.3, blind. Ranking committed BEFORE unseal.

## Blind result vs async prediction

- Bradley (feel, best→worst): **C4 > C2 > C3**  (g2 > g1 > g3)
- Async rung prediction:      **C2 > C4 > C3**

**Verdict: async rung PARTIALLY validated as human-experience proxy.**
The categorical gap reproduced — C3 (the x12 dilution regression) ranked
last blind, with the flaw sightings to match. The C2/C4 order swapped,
which is consistent with round-2's own conclusion that C4 is
"convC2-minus-epsilon": an epsilon gap is exactly what a single blind
game cannot resolve. No crown implication: convC2 retains (round-2
decision stands — C4 also carries the 5.447 val tax).

## Per-game (analyze_behavior, bot port 1)

| game | policy | secs | stk_lost | SD | KO | stk_taken | dmg | notes (blind) |
|------|--------|------|----------|----|----|-----------|-----|---------------|
| g1 | convC2 | 100.5 | 4 | 1 | 3 | 0 | 93.5 | fun, taunted; shield-drop offstage → airdodge SD |
| g2 | convC4 | 100.9 | 4 | 2 | 2 | 0 | 60.4 | jab→multijab repeat loop (possibly benign) |
| g3 | convC3 | 89.2 | 4 | 1 | 3 | 2 | 104.2 | shine grab (!); side-B SD; laser camping; some shield-grab |

(SD/KO columns re-scored 2026-08-12 evening with the trajectory
classifier — the original percent<20 run scored g1/g2 as SD=0. The g1
witnessed airdodge SD now registers, and g2 turns out to have had TWO
untouched deaths hiding as KOs — another point against C4 the blind feel
didn't surface.)

Note the anti-correlation: C3 had the best offense of the session (only
policy to take stocks, most damage) yet felt worst — human "feel" tracks
flaw sightings, not damage. Consistent with the async rung's diversity/
death metrics mattering as much as offense.

## Classifier bug evidence (task #1 debt) — RESOLVED SAME NIGHT

Bradley WITNESSED an airdodge SD in g1, but analyze_behavior scored g1 as
SD=0 (all 4 deaths counted KO). Concrete instance of GOTCHA #94
(percent<20 death heuristic wrong both ways). The trajectory-classifier
backport (hit-since-last-ground/ledge flag, mirroring edge_snippet_mine)
landed in Melee.GameEvents + analyze_behavior that evening; re-scoring
this session's games flipped g1 0→1 SD (the witnessed one) and g2 0→2.
The g1 replay (Game_20260812T193406.slp) is the validation case.

## New flaw-list entries (for bookmark/scenario mining)

- shield-drop offstage → panic airdodge SD (g1/convC2) — SD class, seen live
- jab→multijab repeat loop (g2/convC4) — unclear if unwanted; watch rate
- confirmed repeats from the crown-decider list: side-B SD, laser camping,
  shield-grab susceptibility (all g3/convC3)
