# Position-dependence probe — RESULTS

States: 24387 frames from 6 live-look replays
(bot port 1), stride 4, situation filter :neutral. Counterfactual opponent placement,
same trunk+heads per policy; AR heads teacher-forced on a fixed neutral
prefix for every variant (prefix confound cancels in the deltas).

## Perception — output movement vs baseline per variant

| policy | variant | mean dP(buttons) | TV(main_x) | TV(main_y) | E[main_x] |
|---|---|---:|---:|---:|---:|
| ARrefit | mirror | 0.026 | 0.134 | 0.130 | -0.080 |
| ARrefit | far | 0.041 | 0.163 | 0.194 | -0.173 |
| ARrefit | close_left | 0.021 | 0.132 | 0.104 | 0.051 |
| ARrefit | close_right | 0.020 | 0.099 | 0.092 | -0.148 |
| INDrefit | mirror | 0.028 | 0.136 | 0.127 | -0.099 |
| INDrefit | far | 0.042 | 0.169 | 0.175 | -0.226 |
| INDrefit | close_left | 0.022 | 0.137 | 0.112 | 0.051 |
| INDrefit | close_right | 0.020 | 0.104 | 0.090 | -0.178 |

## Differentiation — summary

| policy | head | states | E[main_x] base | approach_delta | P(z) near | P(z) far |
|---|---|---:|---:|---:|---:|---:|
| ARrefit | autoregressive | 2500 | -0.072 | **-0.199** | 0.309 | 0.274 |
| INDrefit | independent | 2500 | -0.091 | **-0.229** | 0.301 | 0.267 |

Reading guide:
- Perception rows ~0 across variants = the network does not READ opponent
  position at all (training/curation lever).
- Perception alive but approach_delta ~0 = it sees position but the learned
  policy doesn't STEER by it (selection lever — same class as the F1
  airdodge finding: option-selection, not perception).
- approach_delta > 0 = steers toward the opponent when placement flips.
- P(z) near vs far: does grab probability track actual grab range?
