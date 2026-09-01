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
| ARrefit | left_15 | 0.018 | 0.093 | 0.084 | -0.021 |
| ARrefit | right_15 | 0.017 | 0.082 | 0.084 | -0.092 |
| ARrefit | left_25 | 0.018 | 0.107 | 0.088 | 0.011 |
| ARrefit | right_25 | 0.016 | 0.085 | 0.077 | -0.116 |
| ARrefit | left_40 | 0.021 | 0.132 | 0.104 | 0.051 |
| ARrefit | right_40 | 0.020 | 0.099 | 0.092 | -0.148 |
| ARrefit | left_60 | 0.024 | 0.155 | 0.126 | 0.074 |
| ARrefit | right_60 | 0.026 | 0.122 | 0.125 | -0.173 |
| ARrefit | left_90 | 0.029 | 0.177 | 0.155 | 0.076 |
| ARrefit | right_90 | 0.034 | 0.147 | 0.164 | -0.186 |
| ARrefit | left_130 | 0.035 | 0.198 | 0.177 | 0.060 |
| ARrefit | right_130 | 0.042 | 0.167 | 0.197 | -0.183 |
| INDrefit | mirror | 0.028 | 0.136 | 0.127 | -0.099 |
| INDrefit | far | 0.042 | 0.169 | 0.175 | -0.226 |
| INDrefit | left_15 | 0.019 | 0.099 | 0.080 | -0.017 |
| INDrefit | right_15 | 0.017 | 0.086 | 0.082 | -0.113 |
| INDrefit | left_25 | 0.019 | 0.115 | 0.088 | 0.018 |
| INDrefit | right_25 | 0.017 | 0.089 | 0.077 | -0.140 |
| INDrefit | left_40 | 0.022 | 0.137 | 0.112 | 0.051 |
| INDrefit | right_40 | 0.020 | 0.104 | 0.090 | -0.178 |
| INDrefit | left_60 | 0.026 | 0.151 | 0.140 | 0.058 |
| INDrefit | right_60 | 0.026 | 0.130 | 0.116 | -0.213 |
| INDrefit | left_90 | 0.032 | 0.167 | 0.170 | 0.041 |
| INDrefit | right_90 | 0.035 | 0.157 | 0.145 | -0.237 |
| INDrefit | left_130 | 0.038 | 0.184 | 0.193 | 0.009 |
| INDrefit | right_130 | 0.043 | 0.175 | 0.174 | -0.238 |

## Differentiation — distance sweep

E[main_x] with the opponent placed at self x − d (left) / self x + d (right);
approach_delta = right − left (>0 steers toward, <0 away). P(z) = mean grab
probability at that placement.

| policy | d | E[main_x] opp-left | E[main_x] opp-right | approach_delta | P(z) |
|---|---:|---:|---:|---:|---:|
| ARrefit | 15 | -0.021 | -0.092 | **-0.072** | 0.329 |
| ARrefit | 25 | 0.011 | -0.116 | **-0.127** | 0.322 |
| ARrefit | 40 | 0.051 | -0.148 | **-0.199** | 0.309 |
| ARrefit | 60 | 0.074 | -0.173 | **-0.247** | 0.296 |
| ARrefit | 90 | 0.076 | -0.186 | **-0.262** | 0.286 |
| ARrefit | 130 | 0.060 | -0.183 | **-0.243** | 0.283 |
| INDrefit | 15 | -0.017 | -0.113 | **-0.096** | 0.317 |
| INDrefit | 25 | 0.018 | -0.140 | **-0.158** | 0.312 |
| INDrefit | 40 | 0.051 | -0.178 | **-0.229** | 0.301 |
| INDrefit | 60 | 0.058 | -0.213 | **-0.271** | 0.290 |
| INDrefit | 90 | 0.041 | -0.237 | **-0.278** | 0.283 |
| INDrefit | 130 | 0.009 | -0.238 | **-0.247** | 0.284 |

| policy | head | states | E[main_x] base | P(z) base | P(z) far |
|---|---|---:|---:|---:|---:|
| ARrefit | autoregressive | 2500 | -0.072 | 0.318 | 0.274 |
| INDrefit | independent | 2500 | -0.091 | 0.308 | 0.267 |

Reading guide:
- Perception rows ~0 across variants = the network does not READ opponent
  position at all (training/curation lever).
- Perception alive but approach_delta ~0 = it sees position but the learned
  policy doesn't STEER by it (selection lever — same class as the F1
  airdodge finding: option-selection, not perception).
- approach_delta > 0 = steers toward the opponent when placement flips.
- P(z) near vs far: does grab probability track actual grab range?

## F3b VERDICT — no approach band at ANY range; retreat GROWS with distance

- **approach_delta is negative at every distance for both heads** — there
  is no range at which the policy steers toward the opponent. The missing
  approach is not a spacing threshold; the behavior simply does not exist.
- **Retreat strengthens with range**: −0.07/−0.10 at d=15 → peaking
  −0.26/−0.28 at d=90, flattening by 130. INVERTED vs threat-response
  spacing (danger is highest close). Consistent with the Fox corpus's
  long-range game being retreat (run away / laser zone) while the bot
  never learned the approach half of neutral.
- **Point-blank is defensive-option mode, not steering**: at d=15 the
  directional signal is weakest while P(z) is highest (0.33) — up close
  it stops steering and reaches for grab/shield. This is the shield-grab
  spam Bradley sees, localized: it fires hardest exactly when the
  opponent is on top of it.
- **P(z) tracks range in the right direction but weakly**: monotone
  0.33 → 0.28 across the sweep (baseline 0.31–0.32). Grab knows about
  proximity; it just never turns off.
- **AR ≈ IND at every distance** — trunk property, again.

Implication for training: curation target #1 should specifically
oversample expert NEUTRAL WINS — approach sequences (dash-dance → grab /
JC upsmash / dash-attack windows) — because the corpus-average neutral
teaches retreat at all ranges. A V-rollout selector (G3b PASSED) is the
complementary decode-side lever: candidate approach actions exist in the
sample set (the heads have nonzero mass on toward-stick + attack), they
are just never the argmax of the learned marginal.
