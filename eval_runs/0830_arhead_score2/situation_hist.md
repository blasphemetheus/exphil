# Situation → next-option histograms (A1) and distribution distance (B2)

Sets: expert: 600 files, 226534 events · AR: 8 files, 1270 events · IND: 8 files, 1072 events · ep10_cpu: 8 files, 1249 events.
Subject port: expert 1, others 1. Situation = labels active on the
frame BEFORE the option fired (`Situations`); option = `Options.events`. Shares are
% of that set's options in that situation. TV = total-variation distance to the
expert's histogram (0 = identical, 1 = disjoint); KL is smoothed (ε=1e-3).
Cells with n < 20 are marked ⚠ / skipped. `_any` = all frames.

## Summary — TV distance to expert, per situation

| situation | expert n | AR TV (n) | IND TV (n) | ep10_cpu TV (n) |
|---|---:|---:|---:|---:|
| _any | 226534 | 0.64 (1270) | 0.61 (1072) | 0.55 (1249) |
| neutral | 157942 | 0.65 (964) | 0.62 (812) | 0.57 (858) |
| approach | 61170 | 0.63 (308) | 0.62 (255) | 0.62 (245) |
| retreat | 37800 | 0.68 (81) | 0.66 (60) | 0.59 (74) |
| advantage | 37713 | 0.62 (256) | 0.61 (192) | 0.49 (302) |
| disadvantage | 9992 | 0.52 (24) | 0.54 (42) | 0.46 (46) |
| conversion_open | 98251 | 0.59 (506) | 0.57 (436) | 0.47 (558) |
| combo_active | 41986 | 0.63 (199) | 0.64 (163) | 0.43 (249) |
| juggle | 8811 | – (7) | – (13) | – (16) |
| tech_chase | 6765 | 0.68 (93) | 0.70 (55) | 0.60 (69) |
| ledge_trap | 987 | – (0) | – (0) | – (4) |
| edgeguard | 28592 | 0.72 (39) | 0.62 (46) | 0.58 (62) |
| shield_pressure_ours | 5869 | – (0) | – (0) | – (0) |
| pummel_throw_decision | 3050 | 0.54 (150) | 0.49 (131) | 0.59 (89) |
| shield_pressure_theirs | 2726 | 0.26 (91) | 0.20 (65) | 0.28 (47) |
| being_edgeguarded | 6446 | 0.62 (106) | 0.59 (48) | 0.40 (36) |
| recovery_low | 7831 | – (14) | – (7) | – (11) |
| recovery_high | 4512 | 0.69 (97) | 0.65 (42) | 0.51 (26) |
| cornered | 18649 | 0.60 (194) | 0.59 (169) | 0.60 (189) |
| edge_danger | 203 | – (4) | – (1) | – (0) |
| offstage | 12343 | 0.62 (111) | 0.65 (49) | 0.46 (37) |
| ledge_hang | 1085 | – (1) | – (0) | – (0) |
| respawn_invincible | 8665 | – (10) | – (16) | – (17) |
| post_kill_neutral | 29324 | – (16) | 0.83 (24) | 0.74 (42) |
| percent_lead | 27351 | 0.70 (134) | 0.72 (45) | 0.61 (215) |
| percent_deficit | 18268 | 0.72 (22) | – (11) | – (8) |

**Mean TV over reported situations:** AR: 0.62 over 17 situations · IND: 0.61 over 17 situations · ep10_cpu: 0.53 over 17 situations

## Per situation

### `_any`

| option | expert (n=226534) | AR (n=1270) | IND (n=1072) | ep10_cpu (n=1249) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 1.9 | 2.2 | 1.2 |
| double_jump | 15.4 | 1.7 | 2.1 | 5.8 |
| aerial | 14.0 | 2.8 | 4.0 | 5.8 |
| special | 9.9 | 12.3 | 12.5 | 21.6 |
| shield_on | 6.2 | 13.7 | 13.3 | 6.7 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 1.7 | 2.3 | 2.0 |
| wavedash | 2.2 | 4.3 | 7.3 | 6.2 |
| grab | 2.2 | 20.6 | 23.0 | 16.3 |
| waveland | 2.2 | 1.4 | 1.1 | 0.9 |
| tilt | 1.9 | 0.3 | 0.2 | 3.0 |
| smash | 1.4 | 4.4 | 3.3 | 3.3 |
| missed_tech | 1.4 | 0.6 | 1.0 | 1.8 |
| throw | 1.2 | 3.8 | 4.7 | 2.0 |
| jab | 0.9 | 3.5 | 4.2 | 3.4 |
| *options / min in situation* | 125.9 | 90.7 | 73.4 | 76.5 |
| **TV vs expert** | – | 0.64 | 0.61 | 0.55 |
| **KL(set‖expert)** | – | 1.26 | 1.11 | 0.95 |

### `neutral`

| option | expert (n=157942) | AR (n=964) | IND (n=812) | ep10_cpu (n=858) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 2.0 | 2.1 | 1.3 |
| aerial | 15.6 | 2.4 | 4.3 | 5.1 |
| double_jump | 15.5 | 1.6 | 2.2 | 6.1 |
| special | 10.1 | 12.7 | 12.4 | 22.0 |
| shield_on | 6.9 | 14.7 | 14.5 | 7.0 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.6 | 1.2 | 2.1 | 1.7 |
| waveland | 2.5 | 1.7 | 1.5 | 1.0 |
| tilt | 2.2 | 0.3 | 0.1 | 3.5 |
| grab | 2.1 | 22.4 | 25.2 | 17.4 |
| wavedash | 2.0 | 4.7 | 8.4 | 7.2 |
| smash | 1.3 | 4.9 | 3.1 | 3.5 |
| jab | 1.0 | 3.3 | 4.3 | 2.3 |
| roll_forward | 0.8 | 6.3 | 4.9 | 3.8 |
| spotdodge | 0.7 | 12.3 | 7.4 | 15.3 |
| *options / min in situation* | 158.0 | 101.7 | 84.1 | 89.3 |
| **TV vs expert** | – | 0.65 | 0.62 | 0.57 |
| **KL(set‖expert)** | – | 1.30 | 1.16 | 1.03 |

### `approach`

| option | expert (n=61170) | AR (n=308) | IND (n=255) | ep10_cpu (n=245) |
|---|---:|---:|---:|---:|
| dash | 25.6 | 2.3 | 3.9 | 2.0 |
| aerial | 21.9 | 2.9 | 3.9 | 3.7 |
| double_jump | 15.9 | 2.9 | 2.7 | 3.3 |
| special | 9.7 | 10.4 | 9.8 | 17.1 |
| shield_on | 6.2 | 12.7 | 11.4 | 7.3 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| waveland | 2.9 | 3.2 | 2.7 | 1.6 |
| airdodge | 2.8 | 1.0 | 1.2 | 1.2 |
| tilt | 2.3 | 0.3 | 0.0 | 3.7 |
| grab | 2.1 | 22.4 | 23.9 | 11.8 |
| wavedash | 1.8 | 7.1 | 12.9 | 13.5 |
| smash | 1.6 | 4.9 | 1.2 | 2.9 |
| dash_attack | 0.9 | 0.3 | 0.0 | 0.0 |
| jab | 0.9 | 2.6 | 5.9 | 4.5 |
| spotdodge | 0.6 | 10.1 | 5.5 | 21.2 |
| *options / min in situation* | 168.1 | 98.6 | 93.2 | 75.8 |
| **TV vs expert** | – | 0.63 | 0.62 | 0.62 |
| **KL(set‖expert)** | – | 1.30 | 1.26 | 1.28 |

### `retreat`

| option | expert (n=37800) | AR (n=81) | IND (n=60) | ep10_cpu (n=74) |
|---|---:|---:|---:|---:|
| dash | 42.2 | 1.2 | 0.0 | 1.4 |
| double_jump | 13.0 | 1.2 | 1.7 | 5.4 |
| special | 10.4 | 14.8 | 8.3 | 33.8 |
| aerial | 9.9 | 1.2 | 8.3 | 10.8 |
| shield_on | 6.4 | 16.0 | 13.3 | 6.8 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| airdodge | 3.3 | 3.7 | 8.3 | 5.4 |
| waveland | 2.6 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.2 | 8.6 | 1.7 | 0.0 |
| tilt | 1.3 | 1.2 | 0.0 | 0.0 |
| grab | 1.1 | 25.9 | 31.7 | 8.1 |
| roll_forward | 0.8 | 1.2 | 3.3 | 6.8 |
| smash | 0.7 | 3.7 | 5.0 | 2.7 |
| jab | 0.6 | 3.7 | 8.3 | 1.4 |
| spotdodge | 0.6 | 7.4 | 1.7 | 10.8 |
| *options / min in situation* | 144.8 | 61.1 | 49.0 | 73.3 |
| **TV vs expert** | – | 0.68 | 0.66 | 0.59 |
| **KL(set‖expert)** | – | 1.55 | 1.63 | 1.10 |

### `advantage`

| option | expert (n=37713) | AR (n=256) | IND (n=192) | ep10_cpu (n=302) |
|---|---:|---:|---:|---:|
| dash | 31.3 | 1.6 | 3.6 | 1.3 |
| double_jump | 20.1 | 2.3 | 2.1 | 7.0 |
| aerial | 12.7 | 2.7 | 3.1 | 9.3 |
| special | 8.1 | 10.9 | 9.9 | 23.2 |
| throw | 7.0 | 18.8 | 26.0 | 8.3 |
| grab | 4.1 | 16.4 | 16.1 | 14.6 |
| wavedash | 3.3 | 2.7 | 5.2 | 5.3 |
| smash | 2.9 | 3.1 | 3.6 | 3.3 |
| shield_on | 2.9 | 10.9 | 9.9 | 5.0 |
| tilt | 2.3 | 0.4 | 0.5 | 2.3 |
| airdodge | 1.3 | 1.2 | 0.0 | 1.7 |
| waveland | 1.2 | 0.8 | 0.0 | 0.7 |
| jab | 0.8 | 3.9 | 4.2 | 6.6 |
| dash_attack | 0.8 | 0.0 | 0.0 | 0.0 |
| dashdance | 0.6 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 137.1 | 91.8 | 67.3 | 65.6 |
| **TV vs expert** | – | 0.62 | 0.61 | 0.49 |
| **KL(set‖expert)** | – | 1.30 | 1.06 | 0.83 |

### `disadvantage`

| option | expert (n=9992) | AR (n=24) | IND (n=42) | ep10_cpu (n=46) |
|---|---:|---:|---:|---:|
| missed_tech | 24.0 | 16.7 | 14.3 | 21.7 |
| shield_on | 16.1 | 8.3 | 9.5 | 10.9 |
| special | 15.6 | 12.5 | 16.7 | 17.4 |
| tech_roll | 13.2 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 13.1 | 0.0 | 0.0 | 0.0 |
| aerial | 5.0 | 12.5 | 0.0 | 0.0 |
| getup_stand | 4.9 | 0.0 | 0.0 | 0.0 |
| getup_attack | 3.9 | 25.0 | 16.7 | 39.1 |
| dash | 1.7 | 0.0 | 0.0 | 0.0 |
| spotdodge | 1.1 | 8.3 | 4.8 | 2.2 |
| airdodge | 0.4 | 8.3 | 9.5 | 4.3 |
| jab | 0.3 | 0.0 | 2.4 | 2.2 |
| tilt | 0.3 | 0.0 | 0.0 | 0.0 |
| smash | 0.2 | 4.2 | 4.8 | 2.2 |
| grab | 0.2 | 4.2 | 21.4 | 0.0 |
| *options / min in situation* | 36.1 | 78.8 | 98.2 | 81.3 |
| **TV vs expert** | – | 0.52 | 0.54 | 0.46 |
| **KL(set‖expert)** | – | 1.02 | 1.54 | 1.01 |

### `conversion_open`

| option | expert (n=98251) | AR (n=506) | IND (n=436) | ep10_cpu (n=558) |
|---|---:|---:|---:|---:|
| dash | 25.7 | 1.4 | 1.4 | 1.6 |
| double_jump | 15.9 | 1.4 | 1.1 | 6.8 |
| aerial | 13.5 | 3.0 | 2.8 | 7.0 |
| special | 10.9 | 11.5 | 14.0 | 24.6 |
| shield_on | 6.7 | 12.3 | 11.7 | 5.2 |
| grab | 3.1 | 18.8 | 22.2 | 14.0 |
| missed_tech | 3.0 | 1.4 | 2.5 | 4.1 |
| throw | 2.6 | 9.1 | 9.9 | 2.7 |
| wavedash | 2.4 | 3.4 | 4.1 | 4.3 |
| tilt | 2.2 | 0.6 | 0.5 | 3.2 |
| smash | 2.0 | 5.3 | 4.1 | 2.2 |
| airdodge | 1.6 | 1.4 | 2.3 | 2.2 |
| waveland | 1.4 | 0.8 | 1.1 | 0.7 |
| tech_in_place | 1.4 | 0.0 | 0.0 | 0.0 |
| tech_roll | 1.3 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 101.0 | 101.6 | 86.7 | 83.7 |
| **TV vs expert** | – | 0.59 | 0.57 | 0.47 |
| **KL(set‖expert)** | – | 1.04 | 0.94 | 0.66 |

### `combo_active`

| option | expert (n=41986) | AR (n=199) | IND (n=163) | ep10_cpu (n=249) |
|---|---:|---:|---:|---:|
| dash | 29.5 | 2.5 | 4.3 | 2.0 |
| double_jump | 21.0 | 2.0 | 1.2 | 8.8 |
| aerial | 13.2 | 2.5 | 3.1 | 12.9 |
| special | 9.8 | 11.1 | 8.0 | 21.7 |
| throw | 6.3 | 24.1 | 30.7 | 10.0 |
| wavedash | 3.3 | 2.0 | 3.1 | 4.0 |
| shield_on | 3.3 | 10.6 | 9.2 | 4.0 |
| grab | 2.8 | 13.1 | 18.4 | 13.3 |
| smash | 2.5 | 3.5 | 4.3 | 3.6 |
| tilt | 2.5 | 0.5 | 0.0 | 3.2 |
| airdodge | 1.5 | 1.0 | 0.0 | 0.8 |
| waveland | 1.2 | 0.0 | 0.0 | 0.4 |
| jab | 0.9 | 5.0 | 3.7 | 4.4 |
| dashdance | 0.7 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.5 | 0.0 | 0.0 | 0.4 |
| *options / min in situation* | 129.3 | 68.7 | 43.5 | 58.0 |
| **TV vs expert** | – | 0.63 | 0.64 | 0.43 |
| **KL(set‖expert)** | – | 1.25 | 1.15 | 0.64 |

### `juggle`

| option | expert (n=8811) | AR (n=7 ⚠) | IND (n=13 ⚠) | ep10_cpu (n=16 ⚠) |
|---|---:|---:|---:|---:|
| dash | 33.7 | 0.0 | 7.7 | 0.0 |
| double_jump | 31.6 | 0.0 | 0.0 | 12.5 |
| aerial | 20.3 | 0.0 | 0.0 | 18.8 |
| special | 3.3 | 14.3 | 15.4 | 12.5 |
| smash | 2.2 | 0.0 | 7.7 | 18.8 |
| tilt | 1.9 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.9 | 0.0 | 23.1 | 6.3 |
| waveland | 1.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 1.2 | 14.3 | 0.0 | 0.0 |
| airdodge | 0.9 | 0.0 | 0.0 | 6.3 |
| dashdance | 0.4 | 0.0 | 0.0 | 0.0 |
| grab | 0.3 | 28.6 | 7.7 | 12.5 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.2 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 28.6 | 0.0 | 6.3 |
| *options / min in situation* | 122.7 | 37.3 | 70.9 | 72.7 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `tech_chase`

| option | expert (n=6765) | AR (n=93) | IND (n=55) | ep10_cpu (n=69) |
|---|---:|---:|---:|---:|
| dash | 39.5 | 0.0 | 0.0 | 0.0 |
| double_jump | 23.3 | 3.2 | 1.8 | 13.0 |
| grab | 12.0 | 25.8 | 30.9 | 23.2 |
| smash | 6.4 | 6.5 | 3.6 | 2.9 |
| tilt | 4.0 | 0.0 | 1.8 | 2.9 |
| shield_on | 3.4 | 16.1 | 18.2 | 5.8 |
| dash_attack | 2.9 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 9.7 | 12.7 | 23.2 |
| jab | 2.6 | 4.3 | 3.6 | 1.4 |
| aerial | 1.5 | 5.4 | 3.6 | 7.2 |
| dashdance | 1.1 | 0.0 | 0.0 | 0.0 |
| throw | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 6.5 | 3.6 | 15.9 |
| roll_backward | 0.1 | 8.6 | 10.9 | 0.0 |
| roll_forward | 0.0 | 14.0 | 9.1 | 4.3 |
| *options / min in situation* | 175.9 | 140.4 | 124.6 | 126.9 |
| **TV vs expert** | – | 0.68 | 0.70 | 0.60 |
| **KL(set‖expert)** | – | 1.74 | 1.63 | 1.47 |

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

| option | expert (n=28592) | AR (n=39) | IND (n=46) | ep10_cpu (n=62) |
|---|---:|---:|---:|---:|
| dash | 38.4 | 2.6 | 8.7 | 4.8 |
| double_jump | 21.4 | 2.6 | 0.0 | 8.1 |
| special | 9.3 | 20.5 | 23.9 | 9.7 |
| aerial | 9.1 | 2.6 | 8.7 | 6.5 |
| wavedash | 4.3 | 7.7 | 13.0 | 4.8 |
| shield_on | 4.2 | 7.7 | 4.3 | 6.5 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| waveland | 2.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.1 | 0.0 | 0.0 | 0.0 |
| tilt | 1.4 | 0.0 | 0.0 | 3.2 |
| smash | 1.2 | 0.0 | 2.2 | 1.6 |
| jab | 0.7 | 10.3 | 6.5 | 12.9 |
| roll_forward | 0.4 | 12.8 | 8.7 | 6.5 |
| roll_backward | 0.4 | 5.1 | 4.3 | 4.8 |
| dash_attack | 0.4 | 0.0 | 0.0 | 1.6 |
| *options / min in situation* | 154.0 | 70.7 | 79.4 | 74.6 |
| **TV vs expert** | – | 0.72 | 0.62 | 0.58 |
| **KL(set‖expert)** | – | 1.82 | 1.37 | 1.65 |

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

| option | expert (n=3050) | AR (n=150) | IND (n=131) | ep10_cpu (n=89) |
|---|---:|---:|---:|---:|
| throw | 85.9 | 32.0 | 38.2 | 28.1 |
| shield_on | 9.9 | 36.7 | 32.8 | 21.3 |
| dash | 1.7 | 4.0 | 3.1 | 1.1 |
| grab | 0.9 | 2.0 | 0.0 | 0.0 |
| spotdodge | 0.6 | 8.0 | 2.3 | 22.5 |
| tilt | 0.4 | 0.7 | 0.0 | 3.4 |
| jab | 0.2 | 2.7 | 6.9 | 7.9 |
| special | 0.2 | 6.7 | 13.7 | 11.2 |
| smash | 0.2 | 7.3 | 3.1 | 4.5 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 87.8 | 67.0 | 57.4 | 29.1 |
| **TV vs expert** | – | 0.54 | 0.49 | 0.59 |
| **KL(set‖expert)** | – | 0.93 | 0.95 | 1.49 |

### `shield_pressure_theirs`

| option | expert (n=2726) | AR (n=91) | IND (n=65) | ep10_cpu (n=47) |
|---|---:|---:|---:|---:|
| grab | 29.0 | 6.6 | 15.4 | 27.7 |
| roll_forward | 24.9 | 24.2 | 21.5 | 10.6 |
| spotdodge | 23.3 | 31.9 | 41.5 | 48.9 |
| roll_backward | 19.8 | 37.4 | 21.5 | 10.6 |
| shield_on | 2.2 | 0.0 | 0.0 | 0.0 |
| dash | 0.3 | 0.0 | 0.0 | 0.0 |
| tech_roll | 0.1 | 0.0 | 0.0 | 0.0 |
| jab | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 0.0 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.0 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| tilt | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 89.4 | 2213.5 | 1857.1 | 829.4 |
| **TV vs expert** | – | 0.26 | 0.20 | 0.28 |
| **KL(set‖expert)** | – | 0.22 | 0.12 | 0.25 |

### `being_edgeguarded`

| option | expert (n=6446) | AR (n=106) | IND (n=48) | ep10_cpu (n=36) |
|---|---:|---:|---:|---:|
| special | 62.1 | 19.8 | 10.4 | 33.3 |
| aerial | 18.8 | 6.6 | 14.6 | 11.1 |
| airdodge | 11.5 | 7.5 | 12.5 | 16.7 |
| dash | 2.2 | 0.0 | 2.1 | 0.0 |
| shield_on | 1.8 | 12.3 | 14.6 | 8.3 |
| double_jump | 1.4 | 0.9 | 0.0 | 2.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 1.9 | 8.3 | 2.8 |
| tilt | 0.2 | 0.0 | 0.0 | 2.8 |
| waveland | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.2 | 14.2 | 2.1 | 2.8 |
| tech_in_place | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.1 | 4.7 | 2.1 | 2.8 |
| grab | 0.1 | 23.6 | 18.8 | 11.1 |
| *options / min in situation* | 36.0 | 70.5 | 37.8 | 36.8 |
| **TV vs expert** | – | 0.62 | 0.59 | 0.40 |
| **KL(set‖expert)** | – | 2.01 | 1.80 | 0.80 |

### `recovery_low`

| option | expert (n=7831) | AR (n=14 ⚠) | IND (n=7 ⚠) | ep10_cpu (n=11 ⚠) |
|---|---:|---:|---:|---:|
| special | 60.7 | 35.7 | 14.3 | 54.5 |
| aerial | 18.4 | 28.6 | 57.1 | 9.1 |
| ledge_getup | 5.3 | 0.0 | 0.0 | 0.0 |
| airdodge | 4.5 | 28.6 | 28.6 | 36.4 |
| ledge_jump | 4.4 | 0.0 | 0.0 | 0.0 |
| ledge_roll | 2.6 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 1.5 | 7.1 | 0.0 | 0.0 |
| dash | 0.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.8 | 0.0 | 0.0 | 0.0 |
| shield_on | 0.4 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 50.2 | 19.9 | 9.3 | 16.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `recovery_high`

| option | expert (n=4512) | AR (n=97) | IND (n=42) | ep10_cpu (n=26) |
|---|---:|---:|---:|---:|
| special | 50.9 | 18.6 | 9.5 | 23.1 |
| aerial | 25.8 | 3.1 | 7.1 | 11.5 |
| airdodge | 11.5 | 4.1 | 11.9 | 7.7 |
| dash | 3.5 | 0.0 | 2.4 | 0.0 |
| shield_on | 2.8 | 13.4 | 16.7 | 11.5 |
| double_jump | 2.6 | 1.0 | 0.0 | 3.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| tilt | 0.4 | 0.0 | 0.0 | 3.8 |
| roll_forward | 0.3 | 2.1 | 9.5 | 3.8 |
| jab | 0.2 | 1.0 | 9.5 | 3.8 |
| smash | 0.2 | 5.2 | 2.4 | 0.0 |
| getup_stand | 0.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.2 | 25.8 | 21.4 | 19.2 |
| tech_in_place | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 30.8 | 109.7 | 74.3 | 77.7 |
| **TV vs expert** | – | 0.69 | 0.65 | 0.51 |
| **KL(set‖expert)** | – | 2.24 | 1.89 | 1.28 |

### `cornered`

| option | expert (n=18649) | AR (n=194) | IND (n=169) | ep10_cpu (n=189) |
|---|---:|---:|---:|---:|
| dash | 37.9 | 1.5 | 3.6 | 1.6 |
| double_jump | 19.3 | 3.6 | 1.8 | 9.0 |
| shield_on | 15.8 | 15.5 | 16.0 | 7.9 |
| grab | 4.1 | 21.6 | 25.4 | 17.5 |
| special | 3.8 | 10.8 | 17.2 | 21.2 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| tilt | 2.8 | 0.5 | 0.0 | 4.8 |
| roll_forward | 2.7 | 8.2 | 7.1 | 4.2 |
| smash | 1.8 | 6.2 | 4.1 | 4.2 |
| throw | 1.7 | 4.1 | 4.1 | 2.6 |
| spotdodge | 1.5 | 15.5 | 9.5 | 18.0 |
| getup_stand | 1.1 | 0.0 | 0.6 | 0.0 |
| aerial | 1.1 | 1.0 | 1.2 | 2.6 |
| roll_backward | 1.0 | 8.8 | 3.0 | 0.5 |
| jab | 0.9 | 1.5 | 4.7 | 2.6 |
| *options / min in situation* | 169.9 | 111.6 | 64.4 | 88.2 |
| **TV vs expert** | – | 0.60 | 0.59 | 0.60 |
| **KL(set‖expert)** | – | 1.07 | 0.99 | 1.03 |

### `edge_danger`

| option | expert (n=203) | AR (n=4 ⚠) | IND (n=1 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash_attack | 38.4 | 0.0 | 0.0 | 0.0 |
| shield_on | 33.5 | 0.0 | 0.0 | 0.0 |
| special | 11.8 | 25.0 | 0.0 | 0.0 |
| smash | 7.9 | 0.0 | 0.0 | 0.0 |
| dash | 4.4 | 0.0 | 0.0 | 0.0 |
| grab | 3.4 | 0.0 | 100.0 | 0.0 |
| roll_forward | 0.5 | 75.0 | 0.0 | 0.0 |
| *options / min in situation* | 29.4 | 3600.0 | 1800.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `offstage`

| option | expert (n=12343) | AR (n=111) | IND (n=49) | ep10_cpu (n=37) |
|---|---:|---:|---:|---:|
| special | 57.1 | 20.7 | 10.2 | 32.4 |
| aerial | 21.1 | 6.3 | 14.3 | 10.8 |
| airdodge | 7.1 | 7.2 | 14.3 | 16.2 |
| ledge_getup | 3.4 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 2.8 | 0.0 | 0.0 | 0.0 |
| dash | 1.8 | 0.0 | 2.0 | 0.0 |
| ledge_roll | 1.6 | 0.0 | 0.0 | 0.0 |
| double_jump | 1.4 | 0.9 | 0.0 | 2.7 |
| shield_on | 1.3 | 11.7 | 14.3 | 8.1 |
| ledge_attack | 0.9 | 0.9 | 0.0 | 0.0 |
| missed_tech | 0.3 | 0.0 | 0.0 | 0.0 |
| tilt | 0.1 | 0.0 | 0.0 | 2.7 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.1 | 1.8 | 8.2 | 2.7 |
| smash | 0.1 | 4.5 | 2.0 | 0.0 |
| *options / min in situation* | 40.8 | 69.9 | 37.2 | 37.0 |
| **TV vs expert** | – | 0.62 | 0.65 | 0.46 |
| **KL(set‖expert)** | – | 2.12 | 1.92 | 1.06 |

### `ledge_hang`

| option | expert (n=1085) | AR (n=1 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| ledge_getup | 38.4 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 31.7 | 0.0 | 0.0 | 0.0 |
| ledge_roll | 18.5 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 10.7 | 100.0 | 0.0 | 0.0 |
| special | 0.5 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 45.8 | 240.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `respawn_invincible`

| option | expert (n=8665) | AR (n=10 ⚠) | IND (n=16 ⚠) | ep10_cpu (n=17 ⚠) |
|---|---:|---:|---:|---:|
| dash | 56.6 | 0.0 | 0.0 | 0.0 |
| double_jump | 12.3 | 0.0 | 0.0 | 0.0 |
| dashdance | 9.3 | 0.0 | 0.0 | 0.0 |
| aerial | 5.8 | 30.0 | 12.5 | 0.0 |
| special | 5.5 | 10.0 | 31.3 | 17.6 |
| wavedash | 4.0 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 40.0 | 25.0 | 17.6 |
| waveland | 2.2 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.4 | 0.0 | 12.5 | 11.8 |
| dash_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| grab | 0.1 | 0.0 | 0.0 | 23.5 |
| jab | 0.1 | 10.0 | 6.3 | 5.9 |
| roll_backward | 0.1 | 0.0 | 0.0 | 17.6 |
| spotdodge | 0.0 | 10.0 | 6.3 | 5.9 |
| ledge_jump | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 142.5 | 23.5 | 33.5 | 38.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `post_kill_neutral`

| option | expert (n=29324) | AR (n=16 ⚠) | IND (n=24) | ep10_cpu (n=42) |
|---|---:|---:|---:|---:|
| dash | 45.8 | 0.0 | 0.0 | 4.8 |
| double_jump | 14.0 | 0.0 | 0.0 | 4.8 |
| special | 7.5 | 12.5 | 25.0 | 11.9 |
| aerial | 6.7 | 0.0 | 0.0 | 0.0 |
| dashdance | 6.5 | 0.0 | 0.0 | 0.0 |
| shield_on | 4.6 | 12.5 | 16.7 | 14.3 |
| airdodge | 4.2 | 0.0 | 0.0 | 0.0 |
| waveland | 4.1 | 0.0 | 0.0 | 0.0 |
| wavedash | 3.3 | 6.3 | 8.3 | 2.4 |
| roll_forward | 0.5 | 6.3 | 4.2 | 4.8 |
| grab | 0.4 | 25.0 | 25.0 | 31.0 |
| roll_backward | 0.4 | 0.0 | 8.3 | 11.9 |
| missed_tech | 0.4 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.3 | 12.5 | 4.2 | 4.8 |
| ledge_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 205.8 | 38.4 | 57.6 | 72.0 |
| **TV vs expert** | – | – | 0.83 | 0.74 |
| **KL(set‖expert)** | – | – | 2.13 | 2.10 |

### `percent_lead`

| option | expert (n=27351) | AR (n=134) | IND (n=45) | ep10_cpu (n=215) |
|---|---:|---:|---:|---:|
| dash | 34.1 | 3.0 | 0.0 | 0.5 |
| double_jump | 17.5 | 0.7 | 2.2 | 5.1 |
| aerial | 14.5 | 0.7 | 2.2 | 6.0 |
| special | 8.7 | 9.7 | 13.3 | 18.6 |
| shield_on | 5.0 | 12.7 | 13.3 | 7.4 |
| dashdance | 2.8 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.5 | 4.5 | 8.9 | 6.5 |
| tilt | 2.0 | 0.0 | 0.0 | 3.3 |
| airdodge | 1.8 | 0.7 | 0.0 | 0.5 |
| waveland | 1.7 | 0.7 | 0.0 | 0.5 |
| grab | 1.7 | 26.9 | 13.3 | 20.0 |
| smash | 1.6 | 6.7 | 8.9 | 4.7 |
| jab | 1.1 | 3.7 | 4.4 | 3.3 |
| throw | 0.9 | 6.7 | 4.4 | 3.7 |
| dash_attack | 0.7 | 0.0 | 0.0 | 0.5 |
| *options / min in situation* | 138.8 | 76.7 | 66.9 | 69.9 |
| **TV vs expert** | – | 0.70 | 0.72 | 0.61 |
| **KL(set‖expert)** | – | 1.56 | 1.43 | 1.19 |

### `percent_deficit`

| option | expert (n=18268) | AR (n=22) | IND (n=11 ⚠) | ep10_cpu (n=8 ⚠) |
|---|---:|---:|---:|---:|
| dash | 27.3 | 0.0 | 0.0 | 0.0 |
| double_jump | 13.0 | 0.0 | 0.0 | 0.0 |
| aerial | 12.9 | 0.0 | 0.0 | 0.0 |
| special | 12.8 | 13.6 | 9.1 | 12.5 |
| shield_on | 8.5 | 13.6 | 18.2 | 0.0 |
| airdodge | 2.9 | 0.0 | 0.0 | 12.5 |
| dashdance | 2.7 | 0.0 | 0.0 | 0.0 |
| waveland | 2.7 | 0.0 | 0.0 | 0.0 |
| grab | 2.6 | 22.7 | 9.1 | 25.0 |
| missed_tech | 2.3 | 0.0 | 9.1 | 0.0 |
| wavedash | 2.0 | 0.0 | 0.0 | 0.0 |
| tilt | 1.4 | 0.0 | 0.0 | 0.0 |
| throw | 1.3 | 4.5 | 9.1 | 12.5 |
| smash | 1.1 | 13.6 | 0.0 | 0.0 |
| tech_in_place | 1.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 97.0 | 59.8 | 84.3 | 47.0 |
| **TV vs expert** | – | 0.72 | – | – |
| **KL(set‖expert)** | – | 1.73 | – | – |

