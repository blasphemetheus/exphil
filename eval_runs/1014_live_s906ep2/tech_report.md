# Tech report

Sets: expert: 346 games, 980.6 min · bot: 30 games, 71.1 min · human: 30 games, 71.1 min.
Tech situation = a landing from a knockdown: tech (in place / forward / backward; wall techs separate) or
a miss (TECH_MISS_UP/DOWN). Toward/away is relative to the opponent. Punished = any hit within 90 f;
cost = damage taken in those 90 f. Comboed = punish_quality's opening detector (quiet 30 f, gap 45 f,
kill 90 f) on the subject's OWN percent. Wavedash = airdodge ≤ 4 f into a landing from a jumpsquat;
waveland = the same without one. Dash→shield = shield started within 5 f of dashing/running.
Out of shield: a jumpsquat within 3 f of leaving shield = jump cancel (jc_*), classed by what follows it
(upsmash / grab / shine / wavedash / aerial within 20 f / plain jump); otherwise shield-grab, roll, spot
dodge, hit (left shield in hitstun) or drop. Pressured = the episode took shield stun. Connected = the
subject dealt damage within 30 f of leaving shield.

## Tech (own knockdown landings)

| metric | expert | bot | human |
|---|---:|---:|---:|
| tech_situations_per_min | 4.174 | 4.779 | 2.249 |
| tech_rate | 0.514 | 0.259 | 0.344 |
| wall_techs_per_min | 0.058 | 0.028 | 0.028 |
| tech_in_place_share | 0.509 | 0.580 | 0.600 |
| tech_toward_share | 0.256 | 0.148 | 0.109 |
| tech_away_share | 0.235 | 0.273 | 0.291 |
| tech_punished_share | 0.504 | 0.443 | 0.236 |
| tech_cost_per | 5.705 | 4.464 | 2.205 |
| miss_punished_share | 0.477 | 0.333 | 0.248 |
| miss_cost_per | 5.755 | 4.047 | 1.615 |
| miss_hit_on_ground_share | 0.053 | 0.032 | 0.038 |
| miss_getup_neutral_share | 0.161 | 0.123 | 0.219 |
| miss_getup_attack_share | 0.124 | 0.163 | 0.086 |
| miss_getup_toward_share | 0.130 | 0.139 | 0.048 |
| miss_getup_away_share | 0.145 | 0.238 | 0.381 |
| miss_lying_frames_mean | 2.185 | 4.591 | 0.752 |

## Tech-chase (opponent's misses)

| metric | expert | bot | human |
|---|---:|---:|---:|
| opp_tech_situations_per_min | 2.128 | 2.249 | 4.779 |
| opp_tech_rate | 0.428 | 0.344 | 0.259 |
| chase_punished_share | 0.483 | 0.248 | 0.333 |
| chase_cost_per | 5.467 | 1.615 | 4.047 |

## Comboed (openings against the subject)

| metric | expert | bot | human |
|---|---:|---:|---:|
| openings_taken_per_min | 12.291 | 13.256 | 9.404 |
| hits_per_opening_taken | 1.420 | 1.367 | 1.701 |
| dmg_per_opening_taken | 10.892 | 10.992 | 9.357 |
| opening_taken_death_share | 0.049 | 0.037 | 0.021 |
| comboed_3plus_share | 0.075 | 0.069 | 0.169 |

## Defensive movement

| metric | expert | bot | human |
|---|---:|---:|---:|
| shields_per_min | 7.895 | 6.396 | 4.386 |
| dash_shields_per_min | 1.365 | 0.394 | 0.956 |
| shield_time_share | 0.023 | 0.023 | 0.011 |
| spotdodges_per_min | 0.741 | 0.787 | 0.197 |
| rolls_per_min | 1.287 | 1.617 | 0.703 |
| wavedashes_per_min | 4.130 | 2.615 | 2.404 |
| wavelands_per_min | 0.769 | 0.098 | 0.281 |

## Out of shield (how every shield episode ended)

| metric | expert | bot | human |
|---|---:|---:|---:|
| oos_exits_per_min | 9.989 | 7.872 | 4.498 |
| oos_pressured_share | 0.217 | 0.266 | 0.163 |
| oos_jc_share | 0.503 | 0.521 | 0.569 |
| oos_connected_share | 0.173 | 0.136 | 0.200 |
| oos_pressured_connected_share | 0.256 | 0.228 | 0.365 |
| oos_jc_upsmash_share | 0.006 | 0.000 | 0.016 |
| oos_jc_grab_share | 0.001 | 0.004 | 0.041 |
| oos_jc_shine_share | 0.008 | 0.009 | 0.003 |
| oos_jc_wavedash_share | 0.048 | 0.037 | 0.025 |
| oos_jc_aerial_share | 0.148 | 0.196 | 0.228 |
| oos_jc_jump_share | 0.293 | 0.275 | 0.256 |
| oos_shield_grab_share | 0.030 | 0.055 | 0.066 |
| oos_roll_share | 0.126 | 0.202 | 0.156 |
| oos_spotdodge_share | 0.055 | 0.086 | 0.044 |
| oos_hit_share | 0.021 | 0.054 | 0.022 |
| oos_drop_share | 0.265 | 0.082 | 0.144 |

## Out of shield, PRESSURED episodes only (took shield stun)

| metric | expert | bot | human |
|---|---:|---:|---:|
| oos_pressured_jc_upsmash_share | 0.014 | 0.000 | 0.019 |
| oos_pressured_jc_grab_share | 0.002 | 0.013 | 0.038 |
| oos_pressured_jc_shine_share | 0.024 | 0.027 | 0.000 |
| oos_pressured_jc_wavedash_share | 0.054 | 0.034 | 0.000 |
| oos_pressured_jc_aerial_share | 0.206 | 0.201 | 0.269 |
| oos_pressured_jc_jump_share | 0.273 | 0.248 | 0.212 |
| oos_pressured_shield_grab_share | 0.050 | 0.101 | 0.115 |
| oos_pressured_roll_share | 0.100 | 0.174 | 0.135 |
| oos_pressured_spotdodge_share | 0.041 | 0.074 | 0.038 |
| oos_pressured_hit_share | 0.021 | 0.027 | 0.000 |
| oos_pressured_drop_share | 0.214 | 0.101 | 0.173 |
