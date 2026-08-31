# Situation → next-option histograms (A1) and distribution distance (B2)

Sets: expert: 600 files, 226534 events · AR: 8 files, 1377 events · IND: 8 files, 984 events · ep10_cpu: 8 files, 1249 events.
Subject port: expert 1, others 1. Situation = labels active on the
frame BEFORE the option fired (`Situations`); option = `Options.events`. Shares are
% of that set's options in that situation. TV = total-variation distance to the
expert's histogram (0 = identical, 1 = disjoint); KL is smoothed (ε=1e-3).
Cells with n < 20 are marked ⚠ / skipped. `_any` = all frames.

## Summary — TV distance to expert, per situation

| situation | expert n | AR TV (n) | IND TV (n) | ep10_cpu TV (n) |
|---|---:|---:|---:|---:|
| _any | 226534 | 0.64 (1377) | 0.62 (984) | 0.55 (1249) |
| neutral | 157942 | 0.66 (1004) | 0.63 (704) | 0.57 (858) |
| approach | 61170 | 0.67 (308) | 0.61 (203) | 0.62 (245) |
| retreat | 37800 | 0.70 (93) | 0.66 (42) | 0.59 (74) |
| advantage | 37713 | 0.61 (296) | 0.60 (216) | 0.49 (302) |
| disadvantage | 9992 | 0.47 (40) | 0.41 (33) | 0.46 (46) |
| conversion_open | 98251 | 0.57 (636) | 0.58 (444) | 0.47 (558) |
| combo_active | 41986 | 0.63 (245) | 0.59 (184) | 0.43 (249) |
| juggle | 8811 | – (12) | – (4) | – (16) |
| tech_chase | 6765 | 0.67 (98) | 0.60 (59) | 0.60 (69) |
| ledge_trap | 987 | – (0) | – (0) | – (4) |
| edgeguard | 28592 | 0.64 (74) | 0.71 (64) | 0.58 (62) |
| shield_pressure_ours | 5869 | – (0) | – (0) | – (0) |
| pummel_throw_decision | 3050 | 0.51 (156) | 0.42 (119) | 0.59 (89) |
| shield_pressure_theirs | 2726 | 0.22 (99) | 0.22 (55) | 0.28 (47) |
| being_edgeguarded | 6446 | 0.57 (76) | 0.44 (60) | 0.40 (36) |
| recovery_low | 7831 | – (11) | – (13) | – (11) |
| recovery_high | 4512 | 0.64 (69) | 0.54 (49) | 0.51 (26) |
| cornered | 18649 | 0.64 (227) | 0.60 (176) | 0.60 (189) |
| edge_danger | 203 | – (1) | – (0) | – (0) |
| offstage | 12343 | 0.58 (80) | 0.48 (62) | 0.46 (37) |
| ledge_hang | 1085 | – (1) | – (0) | – (0) |
| respawn_invincible | 8665 | – (10) | 0.77 (21) | – (17) |
| post_kill_neutral | 29324 | 0.86 (31) | 0.73 (29) | 0.74 (42) |
| percent_lead | 27351 | 0.69 (353) | 0.62 (64) | 0.61 (215) |
| percent_deficit | 18268 | 0.69 (34) | – (0) | – (8) |

**Mean TV over reported situations:** AR: 0.61 over 18 situations · IND: 0.57 over 18 situations · ep10_cpu: 0.53 over 17 situations

## Per situation

### `_any`

| option | expert (n=226534) | AR (n=1377) | IND (n=984) | ep10_cpu (n=1249) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 1.5 | 1.5 | 1.2 |
| double_jump | 15.4 | 2.0 | 2.2 | 5.8 |
| aerial | 14.0 | 3.3 | 4.4 | 5.8 |
| special | 9.9 | 12.8 | 14.5 | 21.6 |
| shield_on | 6.2 | 13.9 | 12.6 | 6.7 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 1.2 | 2.2 | 2.0 |
| wavedash | 2.2 | 4.5 | 6.3 | 6.2 |
| grab | 2.2 | 22.1 | 22.8 | 16.3 |
| waveland | 2.2 | 1.0 | 0.7 | 0.9 |
| tilt | 1.9 | 0.1 | 0.4 | 3.0 |
| smash | 1.4 | 4.3 | 3.3 | 3.3 |
| missed_tech | 1.4 | 1.2 | 1.0 | 1.8 |
| throw | 1.2 | 4.0 | 5.3 | 2.0 |
| jab | 0.9 | 3.8 | 4.1 | 3.4 |
| *options / min in situation* | 125.9 | 86.1 | 72.0 | 76.5 |
| **TV vs expert** | – | 0.64 | 0.62 | 0.55 |
| **KL(set‖expert)** | – | 1.23 | 1.11 | 0.95 |

### `neutral`

| option | expert (n=157942) | AR (n=1004) | IND (n=704) | ep10_cpu (n=858) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 1.4 | 1.6 | 1.3 |
| aerial | 15.6 | 2.6 | 4.3 | 5.1 |
| double_jump | 15.5 | 2.0 | 2.0 | 6.1 |
| special | 10.1 | 13.0 | 15.1 | 22.0 |
| shield_on | 6.9 | 14.9 | 14.2 | 7.0 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.6 | 1.3 | 1.8 | 1.7 |
| waveland | 2.5 | 1.3 | 1.0 | 1.0 |
| tilt | 2.2 | 0.2 | 0.3 | 3.5 |
| grab | 2.1 | 23.4 | 25.9 | 17.4 |
| wavedash | 2.0 | 5.4 | 6.8 | 7.2 |
| smash | 1.3 | 4.7 | 3.6 | 3.5 |
| jab | 1.0 | 4.7 | 4.1 | 2.3 |
| roll_forward | 0.8 | 5.2 | 6.3 | 3.8 |
| spotdodge | 0.7 | 12.3 | 7.0 | 15.3 |
| *options / min in situation* | 158.0 | 100.3 | 79.9 | 89.3 |
| **TV vs expert** | – | 0.66 | 0.63 | 0.57 |
| **KL(set‖expert)** | – | 1.29 | 1.17 | 1.03 |

### `approach`

| option | expert (n=61170) | AR (n=308) | IND (n=203) | ep10_cpu (n=245) |
|---|---:|---:|---:|---:|
| dash | 25.6 | 1.0 | 2.5 | 2.0 |
| aerial | 21.9 | 2.6 | 6.4 | 3.7 |
| double_jump | 15.9 | 1.9 | 1.5 | 3.3 |
| special | 9.7 | 11.4 | 11.8 | 17.1 |
| shield_on | 6.2 | 11.4 | 12.3 | 7.3 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| waveland | 2.9 | 2.3 | 2.0 | 1.6 |
| airdodge | 2.8 | 1.0 | 2.5 | 1.2 |
| tilt | 2.3 | 0.0 | 0.5 | 3.7 |
| grab | 2.1 | 26.9 | 26.1 | 11.8 |
| wavedash | 1.8 | 7.1 | 9.9 | 13.5 |
| smash | 1.6 | 4.5 | 2.0 | 2.9 |
| dash_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| jab | 0.9 | 3.9 | 3.9 | 4.5 |
| spotdodge | 0.6 | 11.4 | 3.9 | 21.2 |
| *options / min in situation* | 168.1 | 96.0 | 73.3 | 75.8 |
| **TV vs expert** | – | 0.67 | 0.61 | 0.62 |
| **KL(set‖expert)** | – | 1.44 | 1.19 | 1.28 |

### `retreat`

| option | expert (n=37800) | AR (n=93) | IND (n=42) | ep10_cpu (n=74) |
|---|---:|---:|---:|---:|
| dash | 42.2 | 0.0 | 0.0 | 1.4 |
| double_jump | 13.0 | 1.1 | 2.4 | 5.4 |
| special | 10.4 | 14.0 | 23.8 | 33.8 |
| aerial | 9.9 | 3.2 | 4.8 | 10.8 |
| shield_on | 6.4 | 22.6 | 11.9 | 6.8 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| airdodge | 3.3 | 2.2 | 2.4 | 5.4 |
| waveland | 2.6 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.2 | 6.5 | 2.4 | 0.0 |
| tilt | 1.3 | 0.0 | 2.4 | 0.0 |
| grab | 1.1 | 19.4 | 26.2 | 8.1 |
| roll_forward | 0.8 | 2.2 | 7.1 | 6.8 |
| smash | 0.7 | 3.2 | 4.8 | 2.7 |
| jab | 0.6 | 4.3 | 4.8 | 1.4 |
| spotdodge | 0.6 | 15.1 | 2.4 | 10.8 |
| *options / min in situation* | 144.8 | 77.4 | 40.1 | 73.3 |
| **TV vs expert** | – | 0.70 | 0.66 | 0.59 |
| **KL(set‖expert)** | – | 1.57 | 1.42 | 1.10 |

### `advantage`

| option | expert (n=37713) | AR (n=296) | IND (n=216) | ep10_cpu (n=302) |
|---|---:|---:|---:|---:|
| dash | 31.3 | 2.4 | 1.4 | 1.3 |
| double_jump | 20.1 | 2.7 | 2.8 | 7.0 |
| aerial | 12.7 | 4.7 | 4.6 | 9.3 |
| special | 8.1 | 12.8 | 13.9 | 23.2 |
| throw | 7.0 | 18.6 | 24.1 | 8.3 |
| grab | 4.1 | 18.6 | 15.3 | 14.6 |
| wavedash | 3.3 | 2.4 | 4.6 | 5.3 |
| smash | 2.9 | 3.7 | 2.3 | 3.3 |
| shield_on | 2.9 | 11.5 | 7.9 | 5.0 |
| tilt | 2.3 | 0.0 | 0.9 | 2.3 |
| airdodge | 1.3 | 0.0 | 1.4 | 1.7 |
| waveland | 1.2 | 0.3 | 0.0 | 0.7 |
| jab | 0.8 | 1.7 | 4.6 | 6.6 |
| dash_attack | 0.8 | 0.3 | 0.0 | 0.0 |
| dashdance | 0.6 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 137.1 | 75.9 | 73.8 | 65.6 |
| **TV vs expert** | – | 0.61 | 0.60 | 0.49 |
| **KL(set‖expert)** | – | 1.15 | 1.07 | 0.83 |

### `disadvantage`

| option | expert (n=9992) | AR (n=40) | IND (n=33) | ep10_cpu (n=46) |
|---|---:|---:|---:|---:|
| missed_tech | 24.0 | 15.0 | 24.2 | 21.7 |
| shield_on | 16.1 | 15.0 | 12.1 | 10.9 |
| special | 15.6 | 12.5 | 18.2 | 17.4 |
| tech_roll | 13.2 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 13.1 | 0.0 | 0.0 | 0.0 |
| aerial | 5.0 | 10.0 | 3.0 | 0.0 |
| getup_stand | 4.9 | 0.0 | 0.0 | 0.0 |
| getup_attack | 3.9 | 25.0 | 27.3 | 39.1 |
| dash | 1.7 | 0.0 | 0.0 | 0.0 |
| spotdodge | 1.1 | 7.5 | 0.0 | 2.2 |
| airdodge | 0.4 | 2.5 | 0.0 | 4.3 |
| jab | 0.3 | 0.0 | 0.0 | 2.2 |
| tilt | 0.3 | 0.0 | 0.0 | 0.0 |
| smash | 0.2 | 0.0 | 6.1 | 2.2 |
| grab | 0.2 | 12.5 | 9.1 | 0.0 |
| *options / min in situation* | 36.1 | 93.4 | 79.5 | 81.3 |
| **TV vs expert** | – | 0.47 | 0.41 | 0.46 |
| **KL(set‖expert)** | – | 1.04 | 0.97 | 1.01 |

### `conversion_open`

| option | expert (n=98251) | AR (n=636) | IND (n=444) | ep10_cpu (n=558) |
|---|---:|---:|---:|---:|
| dash | 25.7 | 1.6 | 1.6 | 1.6 |
| double_jump | 15.9 | 2.4 | 2.0 | 6.8 |
| aerial | 13.5 | 3.1 | 3.4 | 7.0 |
| special | 10.9 | 11.9 | 18.5 | 24.6 |
| shield_on | 6.7 | 13.2 | 12.4 | 5.2 |
| grab | 3.1 | 20.3 | 19.6 | 14.0 |
| missed_tech | 3.0 | 2.7 | 2.3 | 4.1 |
| throw | 2.6 | 8.5 | 11.0 | 2.7 |
| wavedash | 2.4 | 2.7 | 3.2 | 4.3 |
| tilt | 2.2 | 0.3 | 0.5 | 3.2 |
| smash | 2.0 | 3.9 | 2.5 | 2.2 |
| airdodge | 1.6 | 1.1 | 1.1 | 2.2 |
| waveland | 1.4 | 0.6 | 0.2 | 0.7 |
| tech_in_place | 1.4 | 0.0 | 0.0 | 0.0 |
| tech_roll | 1.3 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 101.0 | 101.7 | 93.2 | 83.7 |
| **TV vs expert** | – | 0.57 | 0.58 | 0.47 |
| **KL(set‖expert)** | – | 0.98 | 0.90 | 0.66 |

### `combo_active`

| option | expert (n=41986) | AR (n=245) | IND (n=184) | ep10_cpu (n=249) |
|---|---:|---:|---:|---:|
| dash | 29.5 | 2.0 | 1.1 | 2.0 |
| double_jump | 21.0 | 2.4 | 3.8 | 8.8 |
| aerial | 13.2 | 3.7 | 4.9 | 12.9 |
| special | 9.8 | 12.7 | 12.5 | 21.7 |
| throw | 6.3 | 22.4 | 28.3 | 10.0 |
| wavedash | 3.3 | 2.0 | 2.7 | 4.0 |
| shield_on | 3.3 | 10.2 | 5.4 | 4.0 |
| grab | 2.8 | 16.7 | 17.9 | 13.3 |
| smash | 2.5 | 3.7 | 2.2 | 3.6 |
| tilt | 2.5 | 0.0 | 0.5 | 3.2 |
| airdodge | 1.5 | 0.4 | 1.6 | 0.8 |
| waveland | 1.2 | 0.0 | 0.0 | 0.4 |
| jab | 0.9 | 2.0 | 3.8 | 4.4 |
| dashdance | 0.7 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.5 | 0.0 | 0.0 | 0.4 |
| *options / min in situation* | 129.3 | 49.6 | 52.9 | 58.0 |
| **TV vs expert** | – | 0.63 | 0.59 | 0.43 |
| **KL(set‖expert)** | – | 1.25 | 1.09 | 0.64 |

### `juggle`

| option | expert (n=8811) | AR (n=12 ⚠) | IND (n=4 ⚠) | ep10_cpu (n=16 ⚠) |
|---|---:|---:|---:|---:|
| dash | 33.7 | 8.3 | 0.0 | 0.0 |
| double_jump | 31.6 | 8.3 | 0.0 | 12.5 |
| aerial | 20.3 | 8.3 | 0.0 | 18.8 |
| special | 3.3 | 8.3 | 0.0 | 12.5 |
| smash | 2.2 | 8.3 | 0.0 | 18.8 |
| tilt | 1.9 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.9 | 8.3 | 0.0 | 6.3 |
| waveland | 1.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 1.2 | 0.0 | 0.0 | 0.0 |
| airdodge | 0.9 | 0.0 | 0.0 | 6.3 |
| dashdance | 0.4 | 0.0 | 0.0 | 0.0 |
| grab | 0.3 | 25.0 | 50.0 | 12.5 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.2 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 8.3 | 0.0 | 6.3 |
| *options / min in situation* | 122.7 | 52.2 | 29.3 | 72.7 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `tech_chase`

| option | expert (n=6765) | AR (n=98) | IND (n=59) | ep10_cpu (n=69) |
|---|---:|---:|---:|---:|
| dash | 39.5 | 3.1 | 3.4 | 0.0 |
| double_jump | 23.3 | 4.1 | 6.8 | 13.0 |
| grab | 12.0 | 29.6 | 22.0 | 23.2 |
| smash | 6.4 | 5.1 | 5.1 | 2.9 |
| tilt | 4.0 | 0.0 | 1.7 | 2.9 |
| shield_on | 3.4 | 14.3 | 11.9 | 5.8 |
| dash_attack | 2.9 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 14.3 | 23.7 | 23.2 |
| jab | 2.6 | 1.0 | 5.1 | 1.4 |
| aerial | 1.5 | 5.1 | 1.7 | 7.2 |
| dashdance | 1.1 | 0.0 | 0.0 | 0.0 |
| throw | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 8.2 | 1.7 | 15.9 |
| roll_backward | 0.1 | 5.1 | 11.9 | 0.0 |
| roll_forward | 0.0 | 10.2 | 5.1 | 4.3 |
| *options / min in situation* | 175.9 | 137.3 | 154.5 | 126.9 |
| **TV vs expert** | – | 0.67 | 0.60 | 0.60 |
| **KL(set‖expert)** | – | 1.46 | 1.30 | 1.47 |

### `ledge_trap`

| option | expert (n=987) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=4 ⚠) |
|---|---:|---:|---:|---:|
| dash | 55.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 21.5 | 0.0 | 0.0 | 50.0 |
| shield_on | 8.1 | 0.0 | 0.0 | 25.0 |
| dashdance | 5.2 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 0.0 | 0.0 | 0.0 |
| tilt | 2.2 | 0.0 | 0.0 | 0.0 |
| jab | 1.4 | 0.0 | 0.0 | 0.0 |
| smash | 1.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.5 | 0.0 | 0.0 | 0.0 |
| aerial | 0.3 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 0.0 | 0.0 | 25.0 |
| roll_backward | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 172.0 | 0.0 | 0.0 | 464.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `edgeguard`

| option | expert (n=28592) | AR (n=74) | IND (n=64) | ep10_cpu (n=62) |
|---|---:|---:|---:|---:|
| dash | 38.4 | 5.4 | 3.1 | 4.8 |
| double_jump | 21.4 | 2.7 | 1.6 | 8.1 |
| special | 9.3 | 14.9 | 20.3 | 9.7 |
| aerial | 9.1 | 4.1 | 1.6 | 6.5 |
| wavedash | 4.3 | 5.4 | 7.8 | 4.8 |
| shield_on | 4.2 | 9.5 | 10.9 | 6.5 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| waveland | 2.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.1 | 1.4 | 0.0 | 0.0 |
| tilt | 1.4 | 1.4 | 1.6 | 3.2 |
| smash | 1.2 | 5.4 | 1.6 | 1.6 |
| jab | 0.7 | 4.1 | 9.4 | 12.9 |
| roll_forward | 0.4 | 6.8 | 10.9 | 6.5 |
| roll_backward | 0.4 | 12.2 | 14.1 | 4.8 |
| dash_attack | 0.4 | 1.4 | 0.0 | 1.6 |
| *options / min in situation* | 154.0 | 101.2 | 69.9 | 74.6 |
| **TV vs expert** | – | 0.64 | 0.71 | 0.58 |
| **KL(set‖expert)** | – | 1.63 | 1.76 | 1.65 |

### `shield_pressure_ours`

| option | expert (n=5869) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| aerial | 23.4 | 0.0 | 0.0 | 0.0 |
| double_jump | 17.0 | 0.0 | 0.0 | 0.0 |
| dash | 15.4 | 0.0 | 0.0 | 0.0 |
| special | 13.0 | 0.0 | 0.0 | 0.0 |
| shield_on | 6.5 | 0.0 | 0.0 | 0.0 |
| grab | 5.9 | 0.0 | 0.0 | 0.0 |
| tilt | 4.5 | 0.0 | 0.0 | 0.0 |
| jab | 2.8 | 0.0 | 0.0 | 0.0 |
| smash | 2.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.0 | 0.0 | 0.0 | 0.0 |
| waveland | 1.4 | 0.0 | 0.0 | 0.0 |
| airdodge | 1.3 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.9 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| dashdance | 0.7 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 197.2 | 0.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `pummel_throw_decision`

| option | expert (n=3050) | AR (n=156) | IND (n=119) | ep10_cpu (n=89) |
|---|---:|---:|---:|---:|
| throw | 85.9 | 35.3 | 43.7 | 28.1 |
| shield_on | 9.9 | 23.1 | 24.4 | 21.3 |
| dash | 1.7 | 1.9 | 1.7 | 1.1 |
| grab | 0.9 | 2.6 | 2.5 | 0.0 |
| spotdodge | 0.6 | 5.8 | 0.8 | 22.5 |
| tilt | 0.4 | 0.6 | 1.7 | 3.4 |
| jab | 0.2 | 9.6 | 8.4 | 7.9 |
| special | 0.2 | 13.5 | 11.8 | 11.2 |
| smash | 0.2 | 7.7 | 5.0 | 4.5 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 87.8 | 60.7 | 55.1 | 29.1 |
| **TV vs expert** | – | 0.51 | 0.42 | 0.59 |
| **KL(set‖expert)** | – | 1.15 | 0.85 | 1.49 |

### `shield_pressure_theirs`

| option | expert (n=2726) | AR (n=99) | IND (n=55) | ep10_cpu (n=47) |
|---|---:|---:|---:|---:|
| grab | 29.0 | 12.1 | 20.0 | 27.7 |
| roll_forward | 24.9 | 22.2 | 14.5 | 10.6 |
| spotdodge | 23.3 | 42.4 | 41.8 | 48.9 |
| roll_backward | 19.8 | 22.2 | 23.6 | 10.6 |
| shield_on | 2.2 | 0.0 | 0.0 | 0.0 |
| dash | 0.3 | 0.0 | 0.0 | 0.0 |
| tech_roll | 0.1 | 0.0 | 0.0 | 0.0 |
| jab | 0.1 | 1.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 0.0 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.0 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| tilt | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 89.4 | 2329.4 | 1833.3 | 829.4 |
| **TV vs expert** | – | 0.22 | 0.22 | 0.28 |
| **KL(set‖expert)** | – | 0.16 | 0.12 | 0.25 |

### `being_edgeguarded`

| option | expert (n=6446) | AR (n=76) | IND (n=60) | ep10_cpu (n=36) |
|---|---:|---:|---:|---:|
| special | 62.1 | 22.4 | 31.7 | 33.3 |
| aerial | 18.8 | 9.2 | 8.3 | 11.1 |
| airdodge | 11.5 | 5.3 | 10.0 | 16.7 |
| dash | 2.2 | 2.6 | 5.0 | 0.0 |
| shield_on | 1.8 | 5.3 | 8.3 | 8.3 |
| double_jump | 1.4 | 1.3 | 1.7 | 2.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 6.6 | 10.0 | 2.8 |
| tilt | 0.2 | 0.0 | 0.0 | 2.8 |
| waveland | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.2 | 7.9 | 1.7 | 2.8 |
| tech_in_place | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.1 | 3.9 | 3.3 | 2.8 |
| grab | 0.1 | 21.1 | 15.0 | 11.1 |
| *options / min in situation* | 36.0 | 63.9 | 42.1 | 36.8 |
| **TV vs expert** | – | 0.57 | 0.44 | 0.40 |
| **KL(set‖expert)** | – | 1.78 | 1.13 | 0.80 |

### `recovery_low`

| option | expert (n=7831) | AR (n=11 ⚠) | IND (n=13 ⚠) | ep10_cpu (n=11 ⚠) |
|---|---:|---:|---:|---:|
| special | 60.7 | 45.5 | 69.2 | 54.5 |
| aerial | 18.4 | 27.3 | 15.4 | 9.1 |
| ledge_getup | 5.3 | 0.0 | 0.0 | 0.0 |
| airdodge | 4.5 | 18.2 | 15.4 | 36.4 |
| ledge_jump | 4.4 | 9.1 | 0.0 | 0.0 |
| ledge_roll | 2.6 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 1.5 | 0.0 | 0.0 | 0.0 |
| dash | 0.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.8 | 0.0 | 0.0 | 0.0 |
| shield_on | 0.4 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 50.2 | 24.2 | 16.0 | 16.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `recovery_high`

| option | expert (n=4512) | AR (n=69) | IND (n=49) | ep10_cpu (n=26) |
|---|---:|---:|---:|---:|
| special | 50.9 | 17.4 | 22.4 | 23.1 |
| aerial | 25.8 | 7.2 | 6.1 | 11.5 |
| airdodge | 11.5 | 2.9 | 8.2 | 7.7 |
| dash | 3.5 | 2.9 | 6.1 | 0.0 |
| shield_on | 2.8 | 5.8 | 10.2 | 11.5 |
| double_jump | 2.6 | 1.4 | 2.0 | 3.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| tilt | 0.4 | 0.0 | 0.0 | 3.8 |
| roll_forward | 0.3 | 7.2 | 12.2 | 3.8 |
| jab | 0.2 | 7.2 | 2.0 | 3.8 |
| smash | 0.2 | 7.2 | 0.0 | 0.0 |
| getup_stand | 0.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.2 | 23.2 | 18.4 | 19.2 |
| tech_in_place | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 30.8 | 84.3 | 73.3 | 77.7 |
| **TV vs expert** | – | 0.64 | 0.54 | 0.51 |
| **KL(set‖expert)** | – | 1.97 | 1.45 | 1.28 |

### `cornered`

| option | expert (n=18649) | AR (n=227) | IND (n=176) | ep10_cpu (n=189) |
|---|---:|---:|---:|---:|
| dash | 37.9 | 0.9 | 1.7 | 1.6 |
| double_jump | 19.3 | 2.6 | 2.3 | 9.0 |
| shield_on | 15.8 | 13.7 | 18.2 | 7.9 |
| grab | 4.1 | 25.6 | 22.7 | 17.5 |
| special | 3.8 | 15.0 | 11.9 | 21.2 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| tilt | 2.8 | 0.0 | 0.6 | 4.8 |
| roll_forward | 2.7 | 6.6 | 7.4 | 4.2 |
| smash | 1.8 | 4.8 | 2.3 | 4.2 |
| throw | 1.7 | 4.8 | 10.2 | 2.6 |
| spotdodge | 1.5 | 13.2 | 9.1 | 18.0 |
| getup_stand | 1.1 | 0.0 | 0.0 | 0.0 |
| aerial | 1.1 | 0.9 | 1.7 | 2.6 |
| roll_backward | 1.0 | 5.3 | 5.7 | 0.5 |
| jab | 0.9 | 5.7 | 4.0 | 2.6 |
| *options / min in situation* | 169.9 | 75.6 | 80.8 | 88.2 |
| **TV vs expert** | – | 0.64 | 0.60 | 0.60 |
| **KL(set‖expert)** | – | 1.15 | 1.01 | 1.03 |

### `edge_danger`

| option | expert (n=203) | AR (n=1 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash_attack | 38.4 | 0.0 | 0.0 | 0.0 |
| shield_on | 33.5 | 0.0 | 0.0 | 0.0 |
| special | 11.8 | 100.0 | 0.0 | 0.0 |
| smash | 7.9 | 0.0 | 0.0 | 0.0 |
| dash | 4.4 | 0.0 | 0.0 | 0.0 |
| grab | 3.4 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.5 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 29.4 | 600.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `offstage`

| option | expert (n=12343) | AR (n=80) | IND (n=62) | ep10_cpu (n=37) |
|---|---:|---:|---:|---:|
| special | 57.1 | 21.3 | 32.3 | 32.4 |
| aerial | 21.1 | 10.0 | 8.1 | 10.8 |
| airdodge | 7.1 | 5.0 | 9.7 | 16.2 |
| ledge_getup | 3.4 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 2.8 | 1.3 | 0.0 | 0.0 |
| dash | 1.8 | 2.5 | 4.8 | 0.0 |
| ledge_roll | 1.6 | 0.0 | 0.0 | 0.0 |
| double_jump | 1.4 | 1.3 | 1.6 | 2.7 |
| shield_on | 1.3 | 5.0 | 8.1 | 8.1 |
| ledge_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.3 | 0.0 | 0.0 | 0.0 |
| tilt | 0.1 | 0.0 | 0.0 | 2.7 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.1 | 6.3 | 9.7 | 2.7 |
| smash | 0.1 | 6.3 | 0.0 | 0.0 |
| *options / min in situation* | 40.8 | 62.9 | 41.8 | 37.0 |
| **TV vs expert** | – | 0.58 | 0.48 | 0.46 |
| **KL(set‖expert)** | – | 1.86 | 1.29 | 1.06 |

### `ledge_hang`

| option | expert (n=1085) | AR (n=1 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| ledge_getup | 38.4 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 31.7 | 100.0 | 0.0 | 0.0 |
| ledge_roll | 18.5 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 10.7 | 0.0 | 0.0 | 0.0 |
| special | 0.5 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 45.8 | 120.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `respawn_invincible`

| option | expert (n=8665) | AR (n=10 ⚠) | IND (n=21) | ep10_cpu (n=17 ⚠) |
|---|---:|---:|---:|---:|
| dash | 56.6 | 0.0 | 0.0 | 0.0 |
| double_jump | 12.3 | 0.0 | 9.5 | 0.0 |
| dashdance | 9.3 | 0.0 | 0.0 | 0.0 |
| aerial | 5.8 | 10.0 | 9.5 | 0.0 |
| special | 5.5 | 10.0 | 0.0 | 17.6 |
| wavedash | 4.0 | 0.0 | 19.0 | 0.0 |
| airdodge | 2.3 | 30.0 | 28.6 | 17.6 |
| waveland | 2.2 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.4 | 10.0 | 4.8 | 11.8 |
| dash_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| grab | 0.1 | 20.0 | 19.0 | 23.5 |
| jab | 0.1 | 0.0 | 0.0 | 5.9 |
| roll_backward | 0.1 | 0.0 | 4.8 | 17.6 |
| spotdodge | 0.0 | 10.0 | 0.0 | 5.9 |
| ledge_jump | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 142.5 | 30.0 | 46.6 | 38.5 |
| **TV vs expert** | – | – | 0.77 | – |
| **KL(set‖expert)** | – | – | 2.28 | – |

### `post_kill_neutral`

| option | expert (n=29324) | AR (n=31) | IND (n=29) | ep10_cpu (n=42) |
|---|---:|---:|---:|---:|
| dash | 45.8 | 0.0 | 3.4 | 4.8 |
| double_jump | 14.0 | 0.0 | 6.9 | 4.8 |
| special | 7.5 | 25.8 | 6.9 | 11.9 |
| aerial | 6.7 | 0.0 | 0.0 | 0.0 |
| dashdance | 6.5 | 0.0 | 0.0 | 0.0 |
| shield_on | 4.6 | 9.7 | 10.3 | 14.3 |
| airdodge | 4.2 | 0.0 | 0.0 | 0.0 |
| waveland | 4.1 | 0.0 | 0.0 | 0.0 |
| wavedash | 3.3 | 0.0 | 17.2 | 2.4 |
| roll_forward | 0.5 | 6.5 | 6.9 | 4.8 |
| grab | 0.4 | 35.5 | 34.5 | 31.0 |
| roll_backward | 0.4 | 6.5 | 6.9 | 11.9 |
| missed_tech | 0.4 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.3 | 12.9 | 0.0 | 4.8 |
| ledge_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 205.8 | 62.0 | 98.8 | 72.0 |
| **TV vs expert** | – | 0.86 | 0.73 | 0.74 |
| **KL(set‖expert)** | – | 2.63 | 2.13 | 2.10 |

### `percent_lead`

| option | expert (n=27351) | AR (n=353) | IND (n=64) | ep10_cpu (n=215) |
|---|---:|---:|---:|---:|
| dash | 34.1 | 2.0 | 4.7 | 0.5 |
| double_jump | 17.5 | 2.8 | 1.6 | 5.1 |
| aerial | 14.5 | 2.3 | 6.3 | 6.0 |
| special | 8.7 | 13.6 | 12.5 | 18.6 |
| shield_on | 5.0 | 13.9 | 12.5 | 7.4 |
| dashdance | 2.8 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.5 | 5.4 | 3.1 | 6.5 |
| tilt | 2.0 | 0.0 | 1.6 | 3.3 |
| airdodge | 1.8 | 0.3 | 3.1 | 0.5 |
| waveland | 1.7 | 0.0 | 0.0 | 0.5 |
| grab | 1.7 | 21.2 | 18.8 | 20.0 |
| smash | 1.6 | 4.2 | 4.7 | 4.7 |
| jab | 1.1 | 2.8 | 0.0 | 3.3 |
| throw | 0.9 | 4.2 | 6.3 | 3.7 |
| dash_attack | 0.7 | 0.0 | 0.0 | 0.5 |
| *options / min in situation* | 138.8 | 91.2 | 75.8 | 69.9 |
| **TV vs expert** | – | 0.69 | 0.62 | 0.61 |
| **KL(set‖expert)** | – | 1.41 | 1.27 | 1.19 |

### `percent_deficit`

| option | expert (n=18268) | AR (n=34) | IND (n=0 ⚠) | ep10_cpu (n=8 ⚠) |
|---|---:|---:|---:|---:|
| dash | 27.3 | 0.0 | 0.0 | 0.0 |
| double_jump | 13.0 | 0.0 | 0.0 | 0.0 |
| aerial | 12.9 | 2.9 | 0.0 | 0.0 |
| special | 12.8 | 8.8 | 0.0 | 12.5 |
| shield_on | 8.5 | 17.6 | 0.0 | 0.0 |
| airdodge | 2.9 | 0.0 | 0.0 | 12.5 |
| dashdance | 2.7 | 0.0 | 0.0 | 0.0 |
| waveland | 2.7 | 0.0 | 0.0 | 0.0 |
| grab | 2.6 | 23.5 | 0.0 | 25.0 |
| missed_tech | 2.3 | 2.9 | 0.0 | 0.0 |
| wavedash | 2.0 | 0.0 | 0.0 | 0.0 |
| tilt | 1.4 | 0.0 | 0.0 | 0.0 |
| throw | 1.3 | 2.9 | 0.0 | 12.5 |
| smash | 1.1 | 5.9 | 0.0 | 0.0 |
| tech_in_place | 1.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 97.0 | 98.4 | 0.0 | 47.0 |
| **TV vs expert** | – | 0.69 | – | – |
| **KL(set‖expert)** | – | 1.48 | – | – |

