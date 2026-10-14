# Tech report

Sets: expert: 346 games, 980.6 min · bot: 30 games, 71.1 min · human: 30 games, 71.1 min.
Tech situation = a landing from a knockdown: tech (in place / forward / backward; wall techs separate) or
a miss (TECH_MISS_UP/DOWN). Toward/away is relative to the opponent. Punished = any hit within 90 f;
cost = damage taken in those 90 f. Comboed = punish_quality's opening detector (quiet 30 f, gap 45 f,
kill 90 f) on the subject's OWN percent. Wavedash = airdodge ≤ 4 f into a landing from a jumpsquat;
waveland = the same without one. Dash→shield = shield started within 5 f of dashing/running.

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
