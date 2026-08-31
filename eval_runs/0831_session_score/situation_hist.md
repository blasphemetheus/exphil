# Situation → next-option histograms (A1) and distribution distance (B2)

Sets: expert: 600 files, 226534 events · AR_human: 13 files, 1427 events · IND_human: 6 files, 581 events · B1_human: 7 files, 505 events.
Subject port: expert 1, others 1. Situation = labels active on the
frame BEFORE the option fired (`Situations`); option = `Options.events`. Shares are
% of that set's options in that situation. TV = total-variation distance to the
expert's histogram (0 = identical, 1 = disjoint); KL is smoothed (ε=1e-3).
Cells with n < 20 are marked ⚠ / skipped. `_any` = all frames.

## Summary — TV distance to expert, per situation

| situation | expert n | AR_human TV (n) | IND_human TV (n) | B1_human TV (n) |
|---|---:|---:|---:|---:|
| _any | 226534 | 0.55 (1427) | 0.54 (581) | 0.57 (505) |
| neutral | 157942 | 0.54 (957) | 0.54 (406) | 0.53 (352) |
| approach | 61170 | 0.54 (302) | 0.51 (118) | 0.59 (95) |
| retreat | 37800 | 0.58 (249) | 0.57 (87) | 0.55 (74) |
| advantage | 37713 | 0.51 (136) | 0.41 (49) | 0.54 (30) |
| disadvantage | 9992 | 0.55 (282) | 0.56 (111) | 0.58 (98) |
| conversion_open | 98251 | 0.54 (716) | 0.47 (236) | 0.55 (275) |
| combo_active | 41986 | 0.49 (135) | 0.34 (47) | 0.53 (34) |
| juggle | 8811 | – (8) | – (2) | – (1) |
| tech_chase | 6765 | 0.62 (47) | – (18) | – (10) |
| ledge_trap | 987 | – (2) | – (1) | – (0) |
| edgeguard | 28592 | 0.57 (80) | 0.80 (22) | – (10) |
| shield_pressure_ours | 5869 | – (3) | – (3) | – (4) |
| pummel_throw_decision | 3050 | 0.52 (41) | 0.66 (30) | 0.73 (21) |
| shield_pressure_theirs | 2726 | 0.20 (41) | 0.14 (40) | 0.09 (21) |
| being_edgeguarded | 6446 | 0.39 (74) | 0.23 (31) | 0.30 (38) |
| recovery_low | 7831 | 0.22 (40) | – (19) | 0.31 (20) |
| recovery_high | 4512 | 0.41 (77) | 0.14 (22) | 0.38 (20) |
| cornered | 18649 | 0.59 (131) | 0.57 (74) | 0.58 (53) |
| edge_danger | 203 | – (0) | – (0) | – (0) |
| offstage | 12343 | 0.35 (117) | 0.24 (41) | 0.29 (40) |
| ledge_hang | 1085 | – (3) | – (2) | – (0) |
| respawn_invincible | 8665 | – (11) | – (8) | – (7) |
| post_kill_neutral | 29324 | 0.66 (30) | – (15) | – (0) |
| percent_lead | 27351 | – (0) | – (0) | – (0) |
| percent_deficit | 18268 | 0.50 (184) | 0.52 (27) | – (12) |

**Mean TV over reported situations:** AR_human: 0.49 over 18 situations · IND_human: 0.45 over 15 situations · B1_human: 0.47 over 14 situations

## Per situation

### `_any`

| option | expert (n=226534) | AR_human (n=1427) | IND_human (n=581) | B1_human (n=505) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 2.1 | 2.8 | 1.6 |
| double_jump | 15.4 | 4.3 | 4.6 | 3.0 |
| aerial | 14.0 | 6.8 | 6.9 | 6.5 |
| special | 9.9 | 19.6 | 12.7 | 17.0 |
| shield_on | 6.2 | 5.2 | 11.2 | 8.9 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 5.4 | 3.3 | 5.7 |
| wavedash | 2.2 | 4.5 | 5.2 | 4.8 |
| grab | 2.2 | 10.4 | 12.4 | 12.9 |
| waveland | 2.2 | 2.0 | 1.4 | 1.8 |
| tilt | 1.9 | 0.8 | 0.5 | 1.6 |
| smash | 1.4 | 3.2 | 1.9 | 1.4 |
| missed_tech | 1.4 | 9.3 | 8.4 | 11.9 |
| throw | 1.2 | 1.0 | 1.0 | 0.6 |
| jab | 0.9 | 3.2 | 2.4 | 2.0 |
| *options / min in situation* | 125.9 | 68.6 | 69.9 | 70.8 |
| **TV vs expert** | – | 0.55 | 0.54 | 0.57 |
| **KL(set‖expert)** | – | 0.94 | 0.96 | 0.99 |

### `neutral`

| option | expert (n=157942) | AR_human (n=957) | IND_human (n=406) | B1_human (n=352) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 2.1 | 3.0 | 2.3 |
| aerial | 15.6 | 6.8 | 7.9 | 7.7 |
| double_jump | 15.5 | 5.2 | 4.4 | 3.7 |
| special | 10.1 | 23.4 | 14.8 | 20.2 |
| shield_on | 6.9 | 6.8 | 14.0 | 10.8 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.6 | 6.7 | 3.9 | 6.5 |
| waveland | 2.5 | 2.5 | 2.0 | 2.3 |
| tilt | 2.2 | 1.3 | 0.5 | 2.3 |
| grab | 2.1 | 11.9 | 14.5 | 15.1 |
| wavedash | 2.0 | 5.7 | 7.1 | 6.3 |
| smash | 1.3 | 2.9 | 1.2 | 1.7 |
| jab | 1.0 | 3.8 | 2.5 | 2.0 |
| roll_forward | 0.8 | 5.7 | 8.4 | 5.1 |
| spotdodge | 0.7 | 10.1 | 9.1 | 8.8 |
| *options / min in situation* | 158.0 | 87.7 | 94.8 | 91.2 |
| **TV vs expert** | – | 0.54 | 0.54 | 0.53 |
| **KL(set‖expert)** | – | 0.83 | 0.88 | 0.83 |

### `approach`

| option | expert (n=61170) | AR_human (n=302) | IND_human (n=118) | B1_human (n=95) |
|---|---:|---:|---:|---:|
| dash | 25.6 | 1.3 | 5.1 | 1.1 |
| aerial | 21.9 | 8.9 | 7.6 | 7.4 |
| double_jump | 15.9 | 4.0 | 5.1 | 1.1 |
| special | 9.7 | 17.9 | 15.3 | 18.9 |
| shield_on | 6.2 | 7.3 | 11.0 | 10.5 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| waveland | 2.9 | 4.3 | 5.1 | 3.2 |
| airdodge | 2.8 | 6.3 | 3.4 | 6.3 |
| tilt | 2.3 | 1.3 | 0.0 | 3.2 |
| grab | 2.1 | 9.6 | 11.0 | 10.5 |
| wavedash | 1.8 | 9.3 | 8.5 | 12.6 |
| smash | 1.6 | 3.3 | 2.5 | 3.2 |
| dash_attack | 0.9 | 0.3 | 0.8 | 0.0 |
| jab | 0.9 | 5.0 | 2.5 | 1.1 |
| spotdodge | 0.6 | 9.3 | 6.8 | 11.6 |
| *options / min in situation* | 168.1 | 78.7 | 85.8 | 71.9 |
| **TV vs expert** | – | 0.54 | 0.51 | 0.59 |
| **KL(set‖expert)** | – | 0.88 | 0.79 | 1.03 |

### `retreat`

| option | expert (n=37800) | AR_human (n=249) | IND_human (n=87) | B1_human (n=74) |
|---|---:|---:|---:|---:|
| dash | 42.2 | 2.0 | 3.4 | 1.4 |
| double_jump | 13.0 | 3.6 | 2.3 | 4.1 |
| special | 10.4 | 29.3 | 21.8 | 25.7 |
| aerial | 9.9 | 8.0 | 11.5 | 12.2 |
| shield_on | 6.4 | 5.6 | 17.2 | 6.8 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| airdodge | 3.3 | 14.1 | 5.7 | 16.2 |
| waveland | 2.6 | 0.4 | 0.0 | 4.1 |
| wavedash | 2.2 | 3.2 | 3.4 | 2.7 |
| tilt | 1.3 | 1.2 | 2.3 | 2.7 |
| grab | 1.1 | 12.4 | 10.3 | 14.9 |
| roll_forward | 0.8 | 4.0 | 5.7 | 1.4 |
| smash | 0.7 | 2.0 | 0.0 | 0.0 |
| jab | 0.6 | 3.2 | 2.3 | 1.4 |
| spotdodge | 0.6 | 7.6 | 9.2 | 6.8 |
| *options / min in situation* | 144.8 | 87.4 | 75.7 | 68.4 |
| **TV vs expert** | – | 0.58 | 0.57 | 0.55 |
| **KL(set‖expert)** | – | 1.00 | 0.94 | 0.98 |

### `advantage`

| option | expert (n=37713) | AR_human (n=136) | IND_human (n=49) | B1_human (n=30) |
|---|---:|---:|---:|---:|
| dash | 31.3 | 3.7 | 4.1 | 0.0 |
| double_jump | 20.1 | 7.4 | 18.4 | 6.7 |
| aerial | 12.7 | 8.1 | 6.1 | 10.0 |
| special | 8.1 | 20.6 | 12.2 | 20.0 |
| throw | 7.0 | 10.3 | 12.2 | 10.0 |
| grab | 4.1 | 16.9 | 18.4 | 16.7 |
| wavedash | 3.3 | 6.6 | 2.0 | 6.7 |
| smash | 2.9 | 1.5 | 6.1 | 0.0 |
| shield_on | 2.9 | 1.5 | 6.1 | 3.3 |
| tilt | 2.3 | 0.0 | 2.0 | 0.0 |
| airdodge | 1.3 | 2.9 | 0.0 | 3.3 |
| waveland | 1.2 | 2.2 | 0.0 | 3.3 |
| jab | 0.8 | 3.7 | 4.1 | 10.0 |
| dash_attack | 0.8 | 0.7 | 0.0 | 0.0 |
| dashdance | 0.6 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 137.1 | 89.6 | 62.4 | 56.8 |
| **TV vs expert** | – | 0.51 | 0.41 | 0.54 |
| **KL(set‖expert)** | – | 0.79 | 0.57 | 0.93 |

### `disadvantage`

| option | expert (n=9992) | AR_human (n=282) | IND_human (n=111) | B1_human (n=98) |
|---|---:|---:|---:|---:|
| missed_tech | 24.0 | 37.9 | 38.7 | 43.9 |
| shield_on | 16.1 | 2.5 | 3.6 | 6.1 |
| special | 15.6 | 7.4 | 5.4 | 5.1 |
| tech_roll | 13.2 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 13.1 | 0.0 | 0.0 | 0.0 |
| aerial | 5.0 | 6.0 | 4.5 | 2.0 |
| getup_stand | 4.9 | 0.0 | 0.0 | 0.0 |
| getup_attack | 3.9 | 34.8 | 39.6 | 32.7 |
| dash | 1.7 | 0.7 | 0.9 | 0.0 |
| spotdodge | 1.1 | 0.4 | 0.9 | 0.0 |
| airdodge | 0.4 | 2.8 | 1.8 | 3.1 |
| jab | 0.3 | 0.7 | 0.9 | 0.0 |
| tilt | 0.3 | 0.0 | 0.0 | 0.0 |
| smash | 0.2 | 5.0 | 1.8 | 1.0 |
| grab | 0.2 | 1.8 | 1.8 | 6.1 |
| *options / min in situation* | 36.1 | 49.3 | 50.1 | 58.7 |
| **TV vs expert** | – | 0.55 | 0.56 | 0.58 |
| **KL(set‖expert)** | – | 1.03 | 1.05 | 1.04 |

### `conversion_open`

| option | expert (n=98251) | AR_human (n=716) | IND_human (n=236) | B1_human (n=275) |
|---|---:|---:|---:|---:|
| dash | 25.7 | 2.0 | 2.5 | 0.4 |
| double_jump | 15.9 | 3.4 | 5.9 | 2.9 |
| aerial | 13.5 | 7.0 | 8.1 | 6.2 |
| special | 10.9 | 20.3 | 12.3 | 15.3 |
| shield_on | 6.7 | 3.1 | 8.1 | 6.5 |
| grab | 3.1 | 8.4 | 10.2 | 12.4 |
| missed_tech | 3.0 | 16.9 | 14.0 | 19.3 |
| throw | 2.6 | 2.0 | 2.5 | 1.1 |
| wavedash | 2.4 | 2.5 | 4.2 | 3.6 |
| tilt | 2.2 | 0.1 | 0.4 | 0.7 |
| smash | 2.0 | 2.8 | 2.5 | 1.1 |
| airdodge | 1.6 | 6.6 | 3.4 | 6.2 |
| waveland | 1.4 | 1.5 | 0.4 | 1.8 |
| tech_in_place | 1.4 | 0.0 | 0.0 | 0.0 |
| tech_roll | 1.3 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 101.0 | 69.7 | 66.9 | 78.0 |
| **TV vs expert** | – | 0.54 | 0.47 | 0.55 |
| **KL(set‖expert)** | – | 0.91 | 0.82 | 0.98 |

### `combo_active`

| option | expert (n=41986) | AR_human (n=135) | IND_human (n=47) | B1_human (n=34) |
|---|---:|---:|---:|---:|
| dash | 29.5 | 5.2 | 2.1 | 0.0 |
| double_jump | 21.0 | 8.1 | 21.3 | 5.9 |
| aerial | 13.2 | 6.7 | 12.8 | 11.8 |
| special | 9.8 | 25.2 | 10.6 | 17.6 |
| throw | 6.3 | 10.4 | 12.8 | 8.8 |
| wavedash | 3.3 | 3.7 | 2.1 | 5.9 |
| shield_on | 3.3 | 3.0 | 4.3 | 2.9 |
| grab | 2.8 | 15.6 | 14.9 | 23.5 |
| smash | 2.5 | 2.2 | 4.3 | 0.0 |
| tilt | 2.5 | 0.0 | 2.1 | 0.0 |
| airdodge | 1.5 | 4.4 | 0.0 | 2.9 |
| waveland | 1.2 | 0.0 | 0.0 | 2.9 |
| jab | 0.9 | 2.2 | 4.3 | 8.8 |
| dashdance | 0.7 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.5 | 0.7 | 0.0 | 0.0 |
| *options / min in situation* | 129.3 | 83.8 | 51.6 | 54.1 |
| **TV vs expert** | – | 0.49 | 0.34 | 0.53 |
| **KL(set‖expert)** | – | 0.68 | 0.57 | 1.01 |

### `juggle`

| option | expert (n=8811) | AR_human (n=8 ⚠) | IND_human (n=2 ⚠) | B1_human (n=1 ⚠) |
|---|---:|---:|---:|---:|
| dash | 33.7 | 0.0 | 0.0 | 0.0 |
| double_jump | 31.6 | 0.0 | 50.0 | 0.0 |
| aerial | 20.3 | 0.0 | 0.0 | 0.0 |
| special | 3.3 | 12.5 | 50.0 | 0.0 |
| smash | 2.2 | 0.0 | 0.0 | 0.0 |
| tilt | 1.9 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.9 | 0.0 | 0.0 | 0.0 |
| waveland | 1.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 1.2 | 12.5 | 0.0 | 0.0 |
| airdodge | 0.9 | 0.0 | 0.0 | 0.0 |
| dashdance | 0.4 | 0.0 | 0.0 | 0.0 |
| grab | 0.3 | 50.0 | 0.0 | 0.0 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.2 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 0.0 | 0.0 | 100.0 |
| *options / min in situation* | 122.7 | 57.0 | 102.9 | 73.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `tech_chase`

| option | expert (n=6765) | AR_human (n=47) | IND_human (n=18 ⚠) | B1_human (n=10 ⚠) |
|---|---:|---:|---:|---:|
| dash | 39.5 | 8.5 | 5.6 | 0.0 |
| double_jump | 23.3 | 10.6 | 33.3 | 20.0 |
| grab | 12.0 | 19.1 | 11.1 | 30.0 |
| smash | 6.4 | 0.0 | 5.6 | 0.0 |
| tilt | 4.0 | 0.0 | 0.0 | 0.0 |
| shield_on | 3.4 | 0.0 | 11.1 | 0.0 |
| dash_attack | 2.9 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 29.8 | 5.6 | 40.0 |
| jab | 2.6 | 6.4 | 11.1 | 0.0 |
| aerial | 1.5 | 2.1 | 5.6 | 10.0 |
| dashdance | 1.1 | 0.0 | 0.0 | 0.0 |
| throw | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 10.6 | 5.6 | 0.0 |
| roll_backward | 0.1 | 10.6 | 0.0 | 0.0 |
| roll_forward | 0.0 | 2.1 | 5.6 | 0.0 |
| *options / min in situation* | 175.9 | 115.4 | 128.8 | 133.8 |
| **TV vs expert** | – | 0.62 | – | – |
| **KL(set‖expert)** | – | 1.49 | – | – |

### `ledge_trap`

| option | expert (n=987) | AR_human (n=2 ⚠) | IND_human (n=1 ⚠) | B1_human (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash | 55.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 21.5 | 0.0 | 100.0 | 0.0 |
| shield_on | 8.1 | 0.0 | 0.0 | 0.0 |
| dashdance | 5.2 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 100.0 | 0.0 | 0.0 |
| tilt | 2.2 | 0.0 | 0.0 | 0.0 |
| jab | 1.4 | 0.0 | 0.0 | 0.0 |
| smash | 1.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.5 | 0.0 | 0.0 | 0.0 |
| aerial | 0.3 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 172.0 | 96.0 | 133.3 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `edgeguard`

| option | expert (n=28592) | AR_human (n=80) | IND_human (n=22) | B1_human (n=10 ⚠) |
|---|---:|---:|---:|---:|
| dash | 38.4 | 1.3 | 0.0 | 10.0 |
| double_jump | 21.4 | 10.0 | 4.5 | 10.0 |
| special | 9.3 | 21.3 | 4.5 | 20.0 |
| aerial | 9.1 | 6.3 | 0.0 | 0.0 |
| wavedash | 4.3 | 6.3 | 4.5 | 10.0 |
| shield_on | 4.2 | 6.3 | 9.1 | 0.0 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| waveland | 2.1 | 3.8 | 0.0 | 0.0 |
| airdodge | 2.1 | 2.5 | 0.0 | 0.0 |
| tilt | 1.4 | 1.3 | 0.0 | 0.0 |
| smash | 1.2 | 0.0 | 0.0 | 0.0 |
| jab | 0.7 | 6.3 | 9.1 | 0.0 |
| roll_forward | 0.4 | 3.8 | 13.6 | 10.0 |
| roll_backward | 0.4 | 3.8 | 9.1 | 0.0 |
| dash_attack | 0.4 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 154.0 | 87.9 | 75.0 | 82.0 |
| **TV vs expert** | – | 0.57 | 0.80 | – |
| **KL(set‖expert)** | – | 1.26 | 2.57 | – |

### `shield_pressure_ours`

| option | expert (n=5869) | AR_human (n=3 ⚠) | IND_human (n=3 ⚠) | B1_human (n=4 ⚠) |
|---|---:|---:|---:|---:|
| aerial | 23.4 | 0.0 | 0.0 | 0.0 |
| double_jump | 17.0 | 0.0 | 0.0 | 25.0 |
| dash | 15.4 | 0.0 | 0.0 | 0.0 |
| special | 13.0 | 33.3 | 33.3 | 25.0 |
| shield_on | 6.5 | 0.0 | 0.0 | 0.0 |
| grab | 5.9 | 0.0 | 0.0 | 25.0 |
| tilt | 4.5 | 33.3 | 0.0 | 0.0 |
| jab | 2.8 | 0.0 | 0.0 | 25.0 |
| smash | 2.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.0 | 33.3 | 0.0 | 0.0 |
| waveland | 1.4 | 0.0 | 0.0 | 0.0 |
| airdodge | 1.3 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.9 | 0.0 | 33.3 | 0.0 |
| dash_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| dashdance | 0.7 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 197.2 | 37.4 | 60.3 | 96.6 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `pummel_throw_decision`

| option | expert (n=3050) | AR_human (n=41) | IND_human (n=30) | B1_human (n=21) |
|---|---:|---:|---:|---:|
| throw | 85.9 | 34.1 | 20.0 | 14.3 |
| shield_on | 9.9 | 26.8 | 33.3 | 33.3 |
| dash | 1.7 | 2.4 | 10.0 | 4.8 |
| grab | 0.9 | 4.9 | 6.7 | 0.0 |
| spotdodge | 0.6 | 7.3 | 13.3 | 14.3 |
| tilt | 0.4 | 0.0 | 0.0 | 0.0 |
| jab | 0.2 | 9.8 | 6.7 | 19.0 |
| special | 0.2 | 9.8 | 6.7 | 4.8 |
| smash | 0.2 | 4.9 | 3.3 | 9.5 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 87.8 | 34.0 | 35.2 | 33.3 |
| **TV vs expert** | – | 0.52 | 0.66 | 0.73 |
| **KL(set‖expert)** | – | 1.04 | 1.31 | 1.87 |

### `shield_pressure_theirs`

| option | expert (n=2726) | AR_human (n=41) | IND_human (n=40) | B1_human (n=21) |
|---|---:|---:|---:|---:|
| grab | 29.0 | 12.2 | 20.0 | 23.8 |
| roll_forward | 24.9 | 29.3 | 30.0 | 23.8 |
| spotdodge | 23.3 | 31.7 | 32.5 | 23.8 |
| roll_backward | 19.8 | 26.8 | 17.5 | 28.6 |
| shield_on | 2.2 | 0.0 | 0.0 | 0.0 |
| dash | 0.3 | 0.0 | 0.0 | 0.0 |
| tech_roll | 0.1 | 0.0 | 0.0 | 0.0 |
| jab | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 0.0 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.0 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| tilt | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 89.4 | 1419.2 | 1285.7 | 720.0 |
| **TV vs expert** | – | 0.20 | 0.14 | 0.09 |
| **KL(set‖expert)** | – | 0.11 | 0.06 | 0.04 |

### `being_edgeguarded`

| option | expert (n=6446) | AR_human (n=74) | IND_human (n=31) | B1_human (n=38) |
|---|---:|---:|---:|---:|
| special | 62.1 | 25.7 | 45.2 | 39.5 |
| aerial | 18.8 | 20.3 | 35.5 | 23.7 |
| airdodge | 11.5 | 31.1 | 16.1 | 31.6 |
| dash | 2.2 | 1.4 | 0.0 | 0.0 |
| shield_on | 1.8 | 1.4 | 3.2 | 0.0 |
| double_jump | 1.4 | 2.7 | 0.0 | 0.0 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 1.4 | 0.0 | 0.0 |
| tilt | 0.2 | 1.4 | 0.0 | 0.0 |
| waveland | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.2 | 4.1 | 0.0 | 0.0 |
| tech_in_place | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.1 | 4.1 | 0.0 | 0.0 |
| grab | 0.1 | 2.7 | 0.0 | 5.3 |
| *options / min in situation* | 36.0 | 33.1 | 29.1 | 38.3 |
| **TV vs expert** | – | 0.39 | 0.23 | 0.30 |
| **KL(set‖expert)** | – | 0.57 | 0.13 | 0.34 |

### `recovery_low`

| option | expert (n=7831) | AR_human (n=40) | IND_human (n=19 ⚠) | B1_human (n=20) |
|---|---:|---:|---:|---:|
| special | 60.7 | 47.5 | 42.1 | 60.0 |
| aerial | 18.4 | 25.0 | 42.1 | 5.0 |
| ledge_getup | 5.3 | 2.5 | 0.0 | 0.0 |
| airdodge | 4.5 | 20.0 | 5.3 | 35.0 |
| ledge_jump | 4.4 | 2.5 | 10.5 | 0.0 |
| ledge_roll | 2.6 | 2.5 | 0.0 | 0.0 |
| ledge_attack | 1.5 | 0.0 | 0.0 | 0.0 |
| dash | 0.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.8 | 0.0 | 0.0 | 0.0 |
| shield_on | 0.4 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 50.2 | 22.5 | 24.0 | 21.5 |
| **TV vs expert** | – | 0.22 | – | 0.31 |
| **KL(set‖expert)** | – | 0.20 | – | 0.60 |

### `recovery_high`

| option | expert (n=4512) | AR_human (n=77) | IND_human (n=22) | B1_human (n=20) |
|---|---:|---:|---:|---:|
| special | 50.9 | 24.7 | 45.5 | 25.0 |
| aerial | 25.8 | 15.6 | 27.3 | 40.0 |
| airdodge | 11.5 | 31.2 | 22.7 | 25.0 |
| dash | 3.5 | 1.3 | 0.0 | 0.0 |
| shield_on | 2.8 | 1.3 | 4.5 | 0.0 |
| double_jump | 2.6 | 2.6 | 0.0 | 0.0 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| tilt | 0.4 | 1.3 | 0.0 | 0.0 |
| roll_forward | 0.3 | 1.3 | 0.0 | 0.0 |
| jab | 0.2 | 5.2 | 0.0 | 0.0 |
| smash | 0.2 | 1.3 | 0.0 | 0.0 |
| getup_stand | 0.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.2 | 3.9 | 0.0 | 10.0 |
| tech_in_place | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 30.8 | 41.2 | 35.0 | 52.2 |
| **TV vs expert** | – | 0.41 | 0.14 | 0.38 |
| **KL(set‖expert)** | – | 0.65 | 0.11 | 0.53 |

### `cornered`

| option | expert (n=18649) | AR_human (n=131) | IND_human (n=74) | B1_human (n=53) |
|---|---:|---:|---:|---:|
| dash | 37.9 | 1.5 | 5.4 | 0.0 |
| double_jump | 19.3 | 8.4 | 4.1 | 7.5 |
| shield_on | 15.8 | 9.9 | 20.3 | 15.1 |
| grab | 4.1 | 11.5 | 14.9 | 17.0 |
| special | 3.8 | 16.0 | 8.1 | 13.2 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| tilt | 2.8 | 2.3 | 0.0 | 1.9 |
| roll_forward | 2.7 | 6.9 | 10.8 | 5.7 |
| smash | 1.8 | 3.8 | 5.4 | 0.0 |
| throw | 1.7 | 0.8 | 0.0 | 3.8 |
| spotdodge | 1.5 | 12.2 | 10.8 | 9.4 |
| getup_stand | 1.1 | 0.0 | 0.0 | 0.0 |
| aerial | 1.1 | 0.8 | 2.7 | 1.9 |
| roll_backward | 1.0 | 6.1 | 4.1 | 5.7 |
| jab | 0.9 | 3.8 | 2.7 | 3.8 |
| *options / min in situation* | 169.9 | 93.8 | 97.5 | 87.8 |
| **TV vs expert** | – | 0.59 | 0.57 | 0.58 |
| **KL(set‖expert)** | – | 1.11 | 0.92 | 1.14 |

### `edge_danger`

| option | expert (n=203) | AR_human (n=0 ⚠) | IND_human (n=0 ⚠) | B1_human (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash_attack | 38.4 | 0.0 | 0.0 | 0.0 |
| shield_on | 33.5 | 0.0 | 0.0 | 0.0 |
| special | 11.8 | 0.0 | 0.0 | 0.0 |
| smash | 7.9 | 0.0 | 0.0 | 0.0 |
| dash | 4.4 | 0.0 | 0.0 | 0.0 |
| grab | 3.4 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.5 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 29.4 | 0.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `offstage`

| option | expert (n=12343) | AR_human (n=117) | IND_human (n=41) | B1_human (n=40) |
|---|---:|---:|---:|---:|
| special | 57.1 | 32.5 | 43.9 | 42.5 |
| aerial | 21.1 | 18.8 | 34.1 | 22.5 |
| airdodge | 7.1 | 27.4 | 14.6 | 30.0 |
| ledge_getup | 3.4 | 0.9 | 0.0 | 0.0 |
| ledge_jump | 2.8 | 0.9 | 4.9 | 0.0 |
| dash | 1.8 | 0.9 | 0.0 | 0.0 |
| ledge_roll | 1.6 | 0.9 | 0.0 | 0.0 |
| double_jump | 1.4 | 1.7 | 0.0 | 0.0 |
| shield_on | 1.3 | 0.9 | 2.4 | 0.0 |
| ledge_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.3 | 0.0 | 0.0 | 0.0 |
| tilt | 0.1 | 0.9 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.1 | 0.9 | 0.0 | 0.0 |
| smash | 0.1 | 0.9 | 0.0 | 0.0 |
| *options / min in situation* | 40.8 | 32.1 | 28.9 | 30.5 |
| **TV vs expert** | – | 0.35 | 0.24 | 0.29 |
| **KL(set‖expert)** | – | 0.53 | 0.17 | 0.46 |

### `ledge_hang`

| option | expert (n=1085) | AR_human (n=3 ⚠) | IND_human (n=2 ⚠) | B1_human (n=0 ⚠) |
|---|---:|---:|---:|---:|
| ledge_getup | 38.4 | 33.3 | 0.0 | 0.0 |
| ledge_jump | 31.7 | 33.3 | 100.0 | 0.0 |
| ledge_roll | 18.5 | 33.3 | 0.0 | 0.0 |
| ledge_attack | 10.7 | 0.0 | 0.0 | 0.0 |
| special | 0.5 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 45.8 | 106.9 | 232.3 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `respawn_invincible`

| option | expert (n=8665) | AR_human (n=11 ⚠) | IND_human (n=8 ⚠) | B1_human (n=7 ⚠) |
|---|---:|---:|---:|---:|
| dash | 56.6 | 0.0 | 12.5 | 0.0 |
| double_jump | 12.3 | 0.0 | 0.0 | 0.0 |
| dashdance | 9.3 | 0.0 | 0.0 | 0.0 |
| aerial | 5.8 | 27.3 | 0.0 | 14.3 |
| special | 5.5 | 27.3 | 25.0 | 57.1 |
| wavedash | 4.0 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 9.1 | 12.5 | 28.6 |
| waveland | 2.2 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.4 | 0.0 | 12.5 | 0.0 |
| dash_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| grab | 0.1 | 18.2 | 12.5 | 0.0 |
| jab | 0.1 | 9.1 | 12.5 | 0.0 |
| roll_backward | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.0 | 9.1 | 0.0 | 0.0 |
| ledge_jump | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 142.5 | 14.4 | 22.8 | 20.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `post_kill_neutral`

| option | expert (n=29324) | AR_human (n=30) | IND_human (n=15 ⚠) | B1_human (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash | 45.8 | 3.3 | 6.7 | 0.0 |
| double_jump | 14.0 | 6.7 | 0.0 | 0.0 |
| special | 7.5 | 26.7 | 6.7 | 0.0 |
| aerial | 6.7 | 3.3 | 6.7 | 0.0 |
| dashdance | 6.5 | 0.0 | 0.0 | 0.0 |
| shield_on | 4.6 | 6.7 | 20.0 | 0.0 |
| airdodge | 4.2 | 0.0 | 0.0 | 0.0 |
| waveland | 4.1 | 3.3 | 0.0 | 0.0 |
| wavedash | 3.3 | 3.3 | 0.0 | 0.0 |
| roll_forward | 0.5 | 3.3 | 6.7 | 0.0 |
| grab | 0.4 | 16.7 | 20.0 | 0.0 |
| roll_backward | 0.4 | 0.0 | 13.3 | 0.0 |
| missed_tech | 0.4 | 3.3 | 0.0 | 0.0 |
| spotdodge | 0.3 | 16.7 | 0.0 | 0.0 |
| ledge_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 205.8 | 57.9 | 90.0 | 0.0 |
| **TV vs expert** | – | 0.66 | – | – |
| **KL(set‖expert)** | – | 1.64 | – | – |

### `percent_lead`

| option | expert (n=27351) | AR_human (n=0 ⚠) | IND_human (n=0 ⚠) | B1_human (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash | 34.1 | 0.0 | 0.0 | 0.0 |
| double_jump | 17.5 | 0.0 | 0.0 | 0.0 |
| aerial | 14.5 | 0.0 | 0.0 | 0.0 |
| special | 8.7 | 0.0 | 0.0 | 0.0 |
| shield_on | 5.0 | 0.0 | 0.0 | 0.0 |
| dashdance | 2.8 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.5 | 0.0 | 0.0 | 0.0 |
| tilt | 2.0 | 0.0 | 0.0 | 0.0 |
| airdodge | 1.8 | 0.0 | 0.0 | 0.0 |
| waveland | 1.7 | 0.0 | 0.0 | 0.0 |
| grab | 1.7 | 0.0 | 0.0 | 0.0 |
| smash | 1.6 | 0.0 | 0.0 | 0.0 |
| jab | 1.1 | 0.0 | 0.0 | 0.0 |
| throw | 0.9 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.7 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 138.8 | 0.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `percent_deficit`

| option | expert (n=18268) | AR_human (n=184) | IND_human (n=27) | B1_human (n=12 ⚠) |
|---|---:|---:|---:|---:|
| dash | 27.3 | 2.2 | 0.0 | 0.0 |
| double_jump | 13.0 | 3.3 | 3.7 | 0.0 |
| aerial | 12.9 | 8.7 | 14.8 | 0.0 |
| special | 12.8 | 19.6 | 14.8 | 25.0 |
| shield_on | 8.5 | 5.4 | 7.4 | 8.3 |
| airdodge | 2.9 | 7.1 | 7.4 | 16.7 |
| dashdance | 2.7 | 0.0 | 0.0 | 0.0 |
| waveland | 2.7 | 1.1 | 0.0 | 8.3 |
| grab | 2.6 | 10.3 | 14.8 | 16.7 |
| missed_tech | 2.3 | 10.9 | 7.4 | 8.3 |
| wavedash | 2.0 | 2.7 | 0.0 | 0.0 |
| tilt | 1.4 | 0.5 | 0.0 | 0.0 |
| throw | 1.3 | 1.1 | 0.0 | 8.3 |
| smash | 1.1 | 1.1 | 3.7 | 0.0 |
| tech_in_place | 1.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 97.0 | 65.7 | 56.4 | 61.6 |
| **TV vs expert** | – | 0.50 | 0.52 | – |
| **KL(set‖expert)** | – | 0.82 | 1.02 | – |

