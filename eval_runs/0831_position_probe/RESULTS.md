# Position-dependence probe — RESULTS

States: 24387 frames from 6 live-look replays
(bot port 1), stride 4. Counterfactual opponent placement,
same trunk+heads per policy; AR heads teacher-forced on a fixed neutral
prefix for every variant (prefix confound cancels in the deltas).

## Perception — output movement vs baseline per variant

| policy | variant | mean dP(buttons) | TV(main_x) | TV(main_y) | E[main_x] |
|---|---|---:|---:|---:|---:|
| ARrefit | mirror | 0.027 | 0.143 | 0.121 | -0.062 |
| ARrefit | far | 0.043 | 0.184 | 0.181 | -0.147 |
| ARrefit | close_left | 0.022 | 0.141 | 0.100 | 0.089 |
| ARrefit | close_right | 0.021 | 0.127 | 0.093 | -0.149 |
| INDrefit | mirror | 0.027 | 0.142 | 0.123 | -0.074 |
| INDrefit | far | 0.042 | 0.185 | 0.172 | -0.176 |
| INDrefit | close_left | 0.022 | 0.135 | 0.114 | 0.080 |
| INDrefit | close_right | 0.022 | 0.127 | 0.097 | -0.161 |

## Differentiation — summary

| policy | head | states | E[main_x] base | approach_delta | P(z) near | P(z) far |
|---|---|---:|---:|---:|---:|---:|
| ARrefit | autoregressive | 2500 | -0.041 | **-0.238** | 0.304 | 0.269 |
| INDrefit | independent | 2500 | -0.046 | **-0.241** | 0.290 | 0.255 |

Reading guide:
- Perception rows ~0 across variants = the network does not READ opponent
  position at all (training/curation lever).
- Perception alive but approach_delta ~0 = it sees position but the learned
  policy doesn't STEER by it (selection lever — same class as the F1
  airdodge finding: option-selection, not perception).
- approach_delta > 0 = steers toward the opponent when placement flips.
- P(z) near vs far: does grab probability track actual grab range?

## VERDICT (both runs: all-states + :neutral-filtered — RESULTS_neutral.md)

Bradley's question "does it actually value where the opponent is?" —
**YES on perception, and the differentiation is systematically RETREAT.**

1. **Perception ALIVE**: every head moves when the opponent is
   counterfactually displaced (TV(main_x) 0.10–0.19, buttons 0.02–0.04
   mean per-button). Not position-blind — curation-for-perception is NOT
   the lever.
2. **It steers AWAY from you**: approach_delta −0.24 all-states, −0.20/−0.23
   neutral-only (opp on left → stick right +0.05, opp on right → stick
   left −0.15..−0.18). The hitstun-DI confound is excluded by the neutral
   filter. It reads your position primarily as a threat vector.
3. **Grab barely tracks range**: P(z) 0.30 near vs 0.27 far (+0.035) —
   and the ABSOLUTE level is the finding: ~30% grab-button probability
   per neutral decision frame. The shield-grab spam Bradley sees is
   mechanistic, position-independent baseline behavior.
4. **Trunk property**: AR and IND heads give near-identical numbers on
   every metric — same class as the no-dash-dance and F2 volume findings.

Mechanistic picture of the live complaint: the trunk registers opponent
position as danger (retreat drift + grab/shield options), never as a
target. "Win positioning / approach / dash-dance into punish" is
advantage-state behavior the policy doesn't express anywhere — feeds
curation target #1 (task 6) and the V-rollout selector (G3b PASSED same
evening): a value head CAN now score approach options a few frames out.

Caveats: FD live-look states only (self-play distribution vs Bradley);
AR probed on a fixed neutral teacher-forced prefix; offset=40 unit
placements may sit inside attack-startup range where retreat is partly
legitimate — a distance-sweep (F3b) would map the retreat-vs-range curve.
