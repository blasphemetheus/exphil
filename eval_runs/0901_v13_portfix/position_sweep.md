# Position-dependence probe — RESULTS

States: 24387 frames from 6 live-look replays
(bot port 1), stride 4, situation filter :neutral. Counterfactual opponent placement,
same trunk+heads per policy; AR heads teacher-forced on a fixed neutral
prefix for every variant (prefix confound cancels in the deltas).

## Perception — output movement vs baseline per variant

| policy | variant | mean dP(buttons) | TV(main_x) | TV(main_y) | E[main_x] |
|---|---|---:|---:|---:|---:|
| v13_ARrefit | mirror | 0.028 | 0.107 | 0.121 | -0.071 |
| v13_ARrefit | far | 0.041 | 0.150 | 0.168 | 0.007 |
| v13_ARrefit | left_15 | 0.018 | 0.073 | 0.076 | -0.067 |
| v13_ARrefit | right_15 | 0.016 | 0.064 | 0.074 | -0.078 |
| v13_ARrefit | left_25 | 0.019 | 0.081 | 0.079 | -0.061 |
| v13_ARrefit | right_25 | 0.016 | 0.066 | 0.071 | -0.077 |
| v13_ARrefit | left_40 | 0.022 | 0.094 | 0.095 | -0.063 |
| v13_ARrefit | right_40 | 0.019 | 0.076 | 0.089 | -0.072 |
| v13_ARrefit | left_60 | 0.027 | 0.115 | 0.121 | -0.095 |
| v13_ARrefit | right_60 | 0.024 | 0.093 | 0.114 | -0.058 |
| v13_ARrefit | left_90 | 0.036 | 0.147 | 0.150 | -0.142 |
| v13_ARrefit | right_90 | 0.032 | 0.121 | 0.145 | -0.028 |
| v13_ARrefit | left_130 | 0.042 | 0.170 | 0.166 | -0.177 |
| v13_ARrefit | right_130 | 0.041 | 0.151 | 0.169 | 0.014 |
| v13_INDrefit | mirror | 0.030 | 0.116 | 0.129 | -0.083 |
| v13_INDrefit | far | 0.044 | 0.161 | 0.175 | 0.005 |
| v13_INDrefit | left_15 | 0.019 | 0.081 | 0.083 | -0.073 |
| v13_INDrefit | right_15 | 0.018 | 0.070 | 0.080 | -0.085 |
| v13_INDrefit | left_25 | 0.020 | 0.089 | 0.087 | -0.065 |
| v13_INDrefit | right_25 | 0.018 | 0.073 | 0.080 | -0.085 |
| v13_INDrefit | left_40 | 0.024 | 0.104 | 0.102 | -0.069 |
| v13_INDrefit | right_40 | 0.021 | 0.082 | 0.096 | -0.081 |
| v13_INDrefit | left_60 | 0.030 | 0.129 | 0.127 | -0.110 |
| v13_INDrefit | right_60 | 0.026 | 0.100 | 0.118 | -0.067 |
| v13_INDrefit | left_90 | 0.037 | 0.166 | 0.157 | -0.169 |
| v13_INDrefit | right_90 | 0.034 | 0.132 | 0.149 | -0.034 |
| v13_INDrefit | left_130 | 0.043 | 0.190 | 0.176 | -0.208 |
| v13_INDrefit | right_130 | 0.043 | 0.164 | 0.175 | 0.013 |

## Differentiation — distance sweep

E[main_x] with the opponent placed at self x − d (left) / self x + d (right);
approach_delta = right − left (>0 steers toward, <0 away). P(z) = mean grab
probability at that placement.

| policy | d | E[main_x] opp-left | E[main_x] opp-right | approach_delta | P(z) |
|---|---:|---:|---:|---:|---:|
| v13_ARrefit | 15 | -0.067 | -0.078 | **-0.011** | 0.260 |
| v13_ARrefit | 25 | -0.061 | -0.077 | **-0.016** | 0.256 |
| v13_ARrefit | 40 | -0.063 | -0.072 | **-0.008** | 0.251 |
| v13_ARrefit | 60 | -0.095 | -0.058 | **0.037** | 0.251 |
| v13_ARrefit | 90 | -0.142 | -0.028 | **0.114** | 0.256 |
| v13_ARrefit | 130 | -0.177 | 0.014 | **0.190** | 0.266 |
| v13_INDrefit | 15 | -0.073 | -0.085 | **-0.012** | 0.243 |
| v13_INDrefit | 25 | -0.065 | -0.085 | **-0.020** | 0.239 |
| v13_INDrefit | 40 | -0.069 | -0.081 | **-0.012** | 0.236 |
| v13_INDrefit | 60 | -0.110 | -0.067 | **0.043** | 0.237 |
| v13_INDrefit | 90 | -0.169 | -0.034 | **0.135** | 0.244 |
| v13_INDrefit | 130 | -0.208 | 0.013 | **0.220** | 0.256 |

| policy | head | states | E[main_x] base | P(z) base | P(z) far |
|---|---|---:|---:|---:|---:|
| v13_ARrefit | autoregressive | 2500 | -0.072 | 0.253 | 0.259 |
| v13_INDrefit | independent | 2500 | -0.081 | 0.238 | 0.243 |

Reading guide:
- Perception rows ~0 across variants = the network does not READ opponent
  position at all (training/curation lever).
- Perception alive but approach_delta ~0 = it sees position but the learned
  policy doesn't STEER by it (selection lever — same class as the F1
  airdodge finding: option-selection, not perception).
- approach_delta > 0 = steers toward the opponent when placement flips.
- P(z) near vs far: does grab probability track actual grab range?
