# Situation → next-option histograms (A1) and distribution distance (B2)

Sets: expert: 600 files, 226534 events · AR: 8 files, 358 events · IND: 8 files, 225 events · ep10_cpu: 8 files, 1249 events.
Subject port: expert 1, others 1. Situation = labels active on the
frame BEFORE the option fired (`Situations`); option = `Options.events`. Shares are
% of that set's options in that situation. TV = total-variation distance to the
expert's histogram (0 = identical, 1 = disjoint); KL is smoothed (ε=1e-3).
Cells with n < 20 are marked ⚠ / skipped. `_any` = all frames.

## Summary — TV distance to expert, per situation

| situation | expert n | AR TV (n) | IND TV (n) | ep10_cpu TV (n) |
|---|---:|---:|---:|---:|
| _any | 226534 | 0.55 (358) | 0.43 (225) | 0.55 (1249) |
| neutral | 157942 | 0.55 (349) | 0.44 (221) | 0.57 (858) |
| approach | 61170 | 0.65 (102) | 0.65 (82) | 0.62 (245) |
| retreat | 37800 | 0.37 (78) | 0.44 (36) | 0.59 (74) |
| advantage | 37713 | – (3) | – (2) | 0.49 (302) |
| disadvantage | 9992 | – (6) | – (2) | 0.46 (46) |
| conversion_open | 98251 | 0.63 (67) | 0.45 (44) | 0.47 (558) |
| combo_active | 41986 | – (10) | – (5) | 0.43 (249) |
| juggle | 8811 | – (0) | – (0) | – (16) |
| tech_chase | 6765 | – (0) | – (0) | 0.60 (69) |
| ledge_trap | 987 | – (0) | – (0) | – (4) |
| edgeguard | 28592 | – (0) | – (0) | 0.58 (62) |
| shield_pressure_ours | 5869 | – (0) | – (0) | – (0) |
| pummel_throw_decision | 3050 | – (0) | – (0) | 0.59 (89) |
| shield_pressure_theirs | 2726 | – (0) | – (0) | 0.28 (47) |
| being_edgeguarded | 6446 | – (3) | – (6) | 0.40 (36) |
| recovery_low | 7831 | – (1) | – (3) | – (11) |
| recovery_high | 4512 | – (2) | – (4) | 0.51 (26) |
| cornered | 18649 | 0.56 (33) | 0.38 (33) | 0.60 (189) |
| edge_danger | 203 | – (2) | – (0) | – (0) |
| offstage | 12343 | – (3) | – (7) | 0.46 (37) |
| ledge_hang | 1085 | – (0) | – (1) | – (0) |
| respawn_invincible | 8665 | – (0) | – (0) | – (17) |
| post_kill_neutral | 29324 | – (0) | – (0) | 0.74 (42) |
| percent_lead | 27351 | – (0) | – (0) | 0.61 (215) |
| percent_deficit | 18268 | – (0) | – (0) | – (8) |

**Mean TV over reported situations:** AR: 0.55 over 5 situations · IND: 0.47 over 5 situations · ep10_cpu: 0.53 over 17 situations

## Per situation

### `_any`

| option | expert (n=226534) | AR (n=358) | IND (n=225) | ep10_cpu (n=1249) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 78.5 | 66.7 | 1.2 |
| double_jump | 15.4 | 5.9 | 19.1 | 5.8 |
| aerial | 14.0 | 0.3 | 1.3 | 5.8 |
| special | 9.9 | 2.5 | 4.0 | 21.6 |
| shield_on | 6.2 | 0.0 | 0.0 | 6.7 |
| dashdance | 2.9 | 10.9 | 0.0 | 0.0 |
| airdodge | 2.3 | 0.0 | 0.0 | 2.0 |
| wavedash | 2.2 | 0.0 | 0.9 | 6.2 |
| grab | 2.2 | 0.0 | 0.0 | 16.3 |
| waveland | 2.2 | 0.0 | 0.0 | 0.9 |
| tilt | 1.9 | 0.0 | 0.0 | 3.0 |
| smash | 1.4 | 0.6 | 1.8 | 3.3 |
| missed_tech | 1.4 | 0.0 | 0.0 | 1.8 |
| throw | 1.2 | 0.0 | 0.0 | 2.0 |
| jab | 0.9 | 0.3 | 0.0 | 3.4 |
| *options / min in situation* | 125.9 | 74.5 | 52.0 | 76.5 |
| **TV vs expert** | – | 0.55 | 0.43 | 0.55 |
| **KL(set‖expert)** | – | 0.68 | 0.50 | 0.95 |

### `neutral`

| option | expert (n=157942) | AR (n=349) | IND (n=221) | ep10_cpu (n=858) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 78.5 | 67.4 | 1.3 |
| aerial | 15.6 | 0.3 | 0.9 | 5.1 |
| double_jump | 15.5 | 6.0 | 19.5 | 6.1 |
| special | 10.1 | 2.6 | 3.6 | 22.0 |
| shield_on | 6.9 | 0.0 | 0.0 | 7.0 |
| dashdance | 3.1 | 11.2 | 0.0 | 0.0 |
| airdodge | 2.6 | 0.0 | 0.0 | 1.7 |
| waveland | 2.5 | 0.0 | 0.0 | 1.0 |
| tilt | 2.2 | 0.0 | 0.0 | 3.5 |
| grab | 2.1 | 0.0 | 0.0 | 17.4 |
| wavedash | 2.0 | 0.0 | 0.9 | 7.2 |
| smash | 1.3 | 0.6 | 1.4 | 3.5 |
| jab | 1.0 | 0.0 | 0.0 | 2.3 |
| roll_forward | 0.8 | 0.0 | 4.5 | 3.8 |
| spotdodge | 0.7 | 0.6 | 0.5 | 15.3 |
| *options / min in situation* | 158.0 | 100.0 | 66.6 | 89.3 |
| **TV vs expert** | – | 0.55 | 0.44 | 0.57 |
| **KL(set‖expert)** | – | 0.69 | 0.51 | 1.03 |

### `approach`

| option | expert (n=61170) | AR (n=102) | IND (n=82) | ep10_cpu (n=245) |
|---|---:|---:|---:|---:|
| dash | 25.6 | 78.4 | 87.8 | 2.0 |
| aerial | 21.9 | 0.0 | 0.0 | 3.7 |
| double_jump | 15.9 | 4.9 | 4.9 | 3.3 |
| special | 9.7 | 0.0 | 2.4 | 17.1 |
| shield_on | 6.2 | 0.0 | 0.0 | 7.3 |
| dashdance | 3.1 | 14.7 | 0.0 | 0.0 |
| waveland | 2.9 | 0.0 | 0.0 | 1.6 |
| airdodge | 2.8 | 0.0 | 0.0 | 1.2 |
| tilt | 2.3 | 0.0 | 0.0 | 3.7 |
| grab | 2.1 | 0.0 | 0.0 | 11.8 |
| wavedash | 1.8 | 0.0 | 2.4 | 13.5 |
| smash | 1.6 | 1.0 | 0.0 | 2.9 |
| dash_attack | 0.9 | 1.0 | 0.0 | 0.0 |
| jab | 0.9 | 0.0 | 0.0 | 4.5 |
| spotdodge | 0.6 | 0.0 | 0.0 | 21.2 |
| *options / min in situation* | 168.1 | 106.5 | 85.8 | 75.8 |
| **TV vs expert** | – | 0.65 | 0.65 | 0.62 |
| **KL(set‖expert)** | – | 0.98 | 0.96 | 1.28 |

### `retreat`

| option | expert (n=37800) | AR (n=78) | IND (n=36) | ep10_cpu (n=74) |
|---|---:|---:|---:|---:|
| dash | 42.2 | 71.8 | 27.8 | 1.4 |
| double_jump | 13.0 | 10.3 | 52.8 | 5.4 |
| special | 10.4 | 5.1 | 8.3 | 33.8 |
| aerial | 9.9 | 1.3 | 5.6 | 10.8 |
| shield_on | 6.4 | 0.0 | 0.0 | 6.8 |
| dashdance | 3.4 | 10.3 | 0.0 | 0.0 |
| airdodge | 3.3 | 0.0 | 0.0 | 5.4 |
| waveland | 2.6 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.2 | 0.0 | 0.0 | 0.0 |
| tilt | 1.3 | 0.0 | 0.0 | 0.0 |
| grab | 1.1 | 0.0 | 0.0 | 8.1 |
| roll_forward | 0.8 | 0.0 | 2.8 | 6.8 |
| smash | 0.7 | 0.0 | 2.8 | 2.7 |
| jab | 0.6 | 0.0 | 0.0 | 1.4 |
| spotdodge | 0.6 | 1.3 | 0.0 | 10.8 |
| *options / min in situation* | 144.8 | 71.1 | 35.6 | 73.3 |
| **TV vs expert** | – | 0.37 | 0.44 | 0.59 |
| **KL(set‖expert)** | – | 0.37 | 0.59 | 1.10 |

### `advantage`

| option | expert (n=37713) | AR (n=3 ⚠) | IND (n=2 ⚠) | ep10_cpu (n=302) |
|---|---:|---:|---:|---:|
| dash | 31.3 | 100.0 | 50.0 | 1.3 |
| double_jump | 20.1 | 0.0 | 0.0 | 7.0 |
| aerial | 12.7 | 0.0 | 50.0 | 9.3 |
| special | 8.1 | 0.0 | 0.0 | 23.2 |
| throw | 7.0 | 0.0 | 0.0 | 8.3 |
| grab | 4.1 | 0.0 | 0.0 | 14.6 |
| wavedash | 3.3 | 0.0 | 0.0 | 5.3 |
| smash | 2.9 | 0.0 | 0.0 | 3.3 |
| shield_on | 2.9 | 0.0 | 0.0 | 5.0 |
| tilt | 2.3 | 0.0 | 0.0 | 2.3 |
| airdodge | 1.3 | 0.0 | 0.0 | 1.7 |
| waveland | 1.2 | 0.0 | 0.0 | 0.7 |
| jab | 0.8 | 0.0 | 0.0 | 6.6 |
| dash_attack | 0.8 | 0.0 | 0.0 | 0.0 |
| dashdance | 0.6 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 137.1 | 67.5 | 128.6 | 65.6 |
| **TV vs expert** | – | – | – | 0.49 |
| **KL(set‖expert)** | – | – | – | 0.83 |

### `disadvantage`

| option | expert (n=9992) | AR (n=6 ⚠) | IND (n=2 ⚠) | ep10_cpu (n=46) |
|---|---:|---:|---:|---:|
| missed_tech | 24.0 | 0.0 | 0.0 | 21.7 |
| shield_on | 16.1 | 0.0 | 0.0 | 10.9 |
| special | 15.6 | 0.0 | 50.0 | 17.4 |
| tech_roll | 13.2 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 13.1 | 16.7 | 0.0 | 0.0 |
| aerial | 5.0 | 0.0 | 0.0 | 0.0 |
| getup_stand | 4.9 | 0.0 | 0.0 | 0.0 |
| getup_attack | 3.9 | 0.0 | 0.0 | 39.1 |
| dash | 1.7 | 66.7 | 0.0 | 0.0 |
| spotdodge | 1.1 | 0.0 | 0.0 | 2.2 |
| airdodge | 0.4 | 0.0 | 0.0 | 4.3 |
| jab | 0.3 | 16.7 | 0.0 | 2.2 |
| tilt | 0.3 | 0.0 | 0.0 | 0.0 |
| smash | 0.2 | 0.0 | 50.0 | 2.2 |
| grab | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 36.1 | 78.5 | 27.6 | 81.3 |
| **TV vs expert** | – | – | – | 0.46 |
| **KL(set‖expert)** | – | – | – | 1.01 |

### `conversion_open`

| option | expert (n=98251) | AR (n=67) | IND (n=44) | ep10_cpu (n=558) |
|---|---:|---:|---:|---:|
| dash | 25.7 | 80.6 | 43.2 | 1.6 |
| double_jump | 15.9 | 1.5 | 20.5 | 6.8 |
| aerial | 13.5 | 0.0 | 4.5 | 7.0 |
| special | 10.9 | 4.5 | 4.5 | 24.6 |
| shield_on | 6.7 | 0.0 | 0.0 | 5.2 |
| grab | 3.1 | 0.0 | 0.0 | 14.0 |
| missed_tech | 3.0 | 0.0 | 0.0 | 4.1 |
| throw | 2.6 | 0.0 | 0.0 | 2.7 |
| wavedash | 2.4 | 0.0 | 0.0 | 4.3 |
| tilt | 2.2 | 0.0 | 0.0 | 3.2 |
| smash | 2.0 | 0.0 | 4.5 | 2.2 |
| airdodge | 1.6 | 0.0 | 0.0 | 2.2 |
| waveland | 1.4 | 0.0 | 0.0 | 0.7 |
| tech_in_place | 1.4 | 1.5 | 0.0 | 0.0 |
| tech_roll | 1.3 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 101.0 | 79.3 | 74.1 | 83.7 |
| **TV vs expert** | – | 0.63 | 0.45 | 0.47 |
| **KL(set‖expert)** | – | 0.93 | 0.75 | 0.66 |

### `combo_active`

| option | expert (n=41986) | AR (n=10 ⚠) | IND (n=5 ⚠) | ep10_cpu (n=249) |
|---|---:|---:|---:|---:|
| dash | 29.5 | 80.0 | 40.0 | 2.0 |
| double_jump | 21.0 | 0.0 | 0.0 | 8.8 |
| aerial | 13.2 | 0.0 | 20.0 | 12.9 |
| special | 9.8 | 0.0 | 0.0 | 21.7 |
| throw | 6.3 | 0.0 | 0.0 | 10.0 |
| wavedash | 3.3 | 0.0 | 0.0 | 4.0 |
| shield_on | 3.3 | 0.0 | 0.0 | 4.0 |
| grab | 2.8 | 0.0 | 0.0 | 13.3 |
| smash | 2.5 | 0.0 | 0.0 | 3.6 |
| tilt | 2.5 | 0.0 | 0.0 | 3.2 |
| airdodge | 1.5 | 0.0 | 0.0 | 0.8 |
| waveland | 1.2 | 0.0 | 0.0 | 0.4 |
| jab | 0.9 | 0.0 | 0.0 | 4.4 |
| dashdance | 0.7 | 10.0 | 0.0 | 0.0 |
| dash_attack | 0.5 | 0.0 | 0.0 | 0.4 |
| *options / min in situation* | 129.3 | 87.6 | 88.7 | 58.0 |
| **TV vs expert** | – | – | – | 0.43 |
| **KL(set‖expert)** | – | – | – | 0.64 |

### `juggle`

| option | expert (n=8811) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=16 ⚠) |
|---|---:|---:|---:|---:|
| dash | 33.7 | 0.0 | 0.0 | 0.0 |
| double_jump | 31.6 | 0.0 | 0.0 | 12.5 |
| aerial | 20.3 | 0.0 | 0.0 | 18.8 |
| special | 3.3 | 0.0 | 0.0 | 12.5 |
| smash | 2.2 | 0.0 | 0.0 | 18.8 |
| tilt | 1.9 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.9 | 0.0 | 0.0 | 6.3 |
| waveland | 1.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 1.2 | 0.0 | 0.0 | 0.0 |
| airdodge | 0.9 | 0.0 | 0.0 | 6.3 |
| dashdance | 0.4 | 0.0 | 0.0 | 0.0 |
| grab | 0.3 | 0.0 | 0.0 | 12.5 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.2 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 0.0 | 0.0 | 6.3 |
| *options / min in situation* | 122.7 | 0.0 | 0.0 | 72.7 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `tech_chase`

| option | expert (n=6765) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=69) |
|---|---:|---:|---:|---:|
| dash | 39.5 | 0.0 | 0.0 | 0.0 |
| double_jump | 23.3 | 0.0 | 0.0 | 13.0 |
| grab | 12.0 | 0.0 | 0.0 | 23.2 |
| smash | 6.4 | 0.0 | 0.0 | 2.9 |
| tilt | 4.0 | 0.0 | 0.0 | 2.9 |
| shield_on | 3.4 | 0.0 | 0.0 | 5.8 |
| dash_attack | 2.9 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 0.0 | 0.0 | 23.2 |
| jab | 2.6 | 0.0 | 0.0 | 1.4 |
| aerial | 1.5 | 0.0 | 0.0 | 7.2 |
| dashdance | 1.1 | 0.0 | 0.0 | 0.0 |
| throw | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 15.9 |
| roll_backward | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.0 | 0.0 | 0.0 | 4.3 |
| *options / min in situation* | 175.9 | 0.0 | 0.0 | 126.9 |
| **TV vs expert** | – | – | – | 0.60 |
| **KL(set‖expert)** | – | – | – | 1.47 |

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

| option | expert (n=28592) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=62) |
|---|---:|---:|---:|---:|
| dash | 38.4 | 0.0 | 0.0 | 4.8 |
| double_jump | 21.4 | 0.0 | 0.0 | 8.1 |
| special | 9.3 | 0.0 | 0.0 | 9.7 |
| aerial | 9.1 | 0.0 | 0.0 | 6.5 |
| wavedash | 4.3 | 0.0 | 0.0 | 4.8 |
| shield_on | 4.2 | 0.0 | 0.0 | 6.5 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| waveland | 2.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.1 | 0.0 | 0.0 | 0.0 |
| tilt | 1.4 | 0.0 | 0.0 | 3.2 |
| smash | 1.2 | 0.0 | 0.0 | 1.6 |
| jab | 0.7 | 0.0 | 0.0 | 12.9 |
| roll_forward | 0.4 | 0.0 | 0.0 | 6.5 |
| roll_backward | 0.4 | 0.0 | 0.0 | 4.8 |
| dash_attack | 0.4 | 0.0 | 0.0 | 1.6 |
| *options / min in situation* | 154.0 | 0.0 | 0.0 | 74.6 |
| **TV vs expert** | – | – | – | 0.58 |
| **KL(set‖expert)** | – | – | – | 1.65 |

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

| option | expert (n=3050) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=89) |
|---|---:|---:|---:|---:|
| throw | 85.9 | 0.0 | 0.0 | 28.1 |
| shield_on | 9.9 | 0.0 | 0.0 | 21.3 |
| dash | 1.7 | 0.0 | 0.0 | 1.1 |
| grab | 0.9 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.6 | 0.0 | 0.0 | 22.5 |
| tilt | 0.4 | 0.0 | 0.0 | 3.4 |
| jab | 0.2 | 0.0 | 0.0 | 7.9 |
| special | 0.2 | 0.0 | 0.0 | 11.2 |
| smash | 0.2 | 0.0 | 0.0 | 4.5 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 87.8 | 0.0 | 0.0 | 29.1 |
| **TV vs expert** | – | – | – | 0.59 |
| **KL(set‖expert)** | – | – | – | 1.49 |

### `shield_pressure_theirs`

| option | expert (n=2726) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=47) |
|---|---:|---:|---:|---:|
| grab | 29.0 | 0.0 | 0.0 | 27.7 |
| roll_forward | 24.9 | 0.0 | 0.0 | 10.6 |
| spotdodge | 23.3 | 0.0 | 0.0 | 48.9 |
| roll_backward | 19.8 | 0.0 | 0.0 | 10.6 |
| shield_on | 2.2 | 0.0 | 0.0 | 0.0 |
| dash | 0.3 | 0.0 | 0.0 | 0.0 |
| tech_roll | 0.1 | 0.0 | 0.0 | 0.0 |
| jab | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 0.0 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.0 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| tilt | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 89.4 | 0.0 | 0.0 | 829.4 |
| **TV vs expert** | – | – | – | 0.28 |
| **KL(set‖expert)** | – | – | – | 0.25 |

### `being_edgeguarded`

| option | expert (n=6446) | AR (n=3 ⚠) | IND (n=6 ⚠) | ep10_cpu (n=36) |
|---|---:|---:|---:|---:|
| special | 62.1 | 33.3 | 33.3 | 33.3 |
| aerial | 18.8 | 0.0 | 0.0 | 11.1 |
| airdodge | 11.5 | 0.0 | 0.0 | 16.7 |
| dash | 2.2 | 66.7 | 50.0 | 0.0 |
| shield_on | 1.8 | 0.0 | 0.0 | 8.3 |
| double_jump | 1.4 | 0.0 | 16.7 | 2.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 0.0 | 0.0 | 2.8 |
| tilt | 0.2 | 0.0 | 0.0 | 2.8 |
| waveland | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.2 | 0.0 | 0.0 | 2.8 |
| tech_in_place | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.1 | 0.0 | 0.0 | 2.8 |
| grab | 0.1 | 0.0 | 0.0 | 11.1 |
| *options / min in situation* | 36.0 | 3.0 | 5.9 | 36.8 |
| **TV vs expert** | – | – | – | 0.40 |
| **KL(set‖expert)** | – | – | – | 0.80 |

### `recovery_low`

| option | expert (n=7831) | AR (n=1 ⚠) | IND (n=3 ⚠) | ep10_cpu (n=11 ⚠) |
|---|---:|---:|---:|---:|
| special | 60.7 | 100.0 | 66.7 | 54.5 |
| aerial | 18.4 | 0.0 | 0.0 | 9.1 |
| ledge_getup | 5.3 | 0.0 | 33.3 | 0.0 |
| airdodge | 4.5 | 0.0 | 0.0 | 36.4 |
| ledge_jump | 4.4 | 0.0 | 0.0 | 0.0 |
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
| *options / min in situation* | 50.2 | 1.1 | 3.2 | 16.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `recovery_high`

| option | expert (n=4512) | AR (n=2 ⚠) | IND (n=4 ⚠) | ep10_cpu (n=26) |
|---|---:|---:|---:|---:|
| special | 50.9 | 0.0 | 0.0 | 23.1 |
| aerial | 25.8 | 0.0 | 0.0 | 11.5 |
| airdodge | 11.5 | 0.0 | 0.0 | 7.7 |
| dash | 3.5 | 100.0 | 75.0 | 0.0 |
| shield_on | 2.8 | 0.0 | 0.0 | 11.5 |
| double_jump | 2.6 | 0.0 | 25.0 | 3.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| tilt | 0.4 | 0.0 | 0.0 | 3.8 |
| roll_forward | 0.3 | 0.0 | 0.0 | 3.8 |
| jab | 0.2 | 0.0 | 0.0 | 3.8 |
| smash | 0.2 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.2 | 0.0 | 0.0 | 19.2 |
| tech_in_place | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 30.8 | 20.1 | 42.7 | 77.7 |
| **TV vs expert** | – | – | – | 0.51 |
| **KL(set‖expert)** | – | – | – | 1.28 |

### `cornered`

| option | expert (n=18649) | AR (n=33) | IND (n=33) | ep10_cpu (n=189) |
|---|---:|---:|---:|---:|
| dash | 37.9 | 90.9 | 60.6 | 1.6 |
| double_jump | 19.3 | 0.0 | 21.2 | 9.0 |
| shield_on | 15.8 | 0.0 | 0.0 | 7.9 |
| grab | 4.1 | 0.0 | 0.0 | 17.5 |
| special | 3.8 | 3.0 | 0.0 | 21.2 |
| dashdance | 2.9 | 6.1 | 0.0 | 0.0 |
| tilt | 2.8 | 0.0 | 0.0 | 4.8 |
| roll_forward | 2.7 | 0.0 | 15.2 | 4.2 |
| smash | 1.8 | 0.0 | 3.0 | 4.2 |
| throw | 1.7 | 0.0 | 0.0 | 2.6 |
| spotdodge | 1.5 | 0.0 | 0.0 | 18.0 |
| getup_stand | 1.1 | 0.0 | 0.0 | 0.0 |
| aerial | 1.1 | 0.0 | 0.0 | 2.6 |
| roll_backward | 1.0 | 0.0 | 0.0 | 0.5 |
| jab | 0.9 | 0.0 | 0.0 | 2.6 |
| *options / min in situation* | 169.9 | 137.2 | 106.5 | 88.2 |
| **TV vs expert** | – | 0.56 | 0.38 | 0.60 |
| **KL(set‖expert)** | – | 0.77 | 0.53 | 1.03 |

### `edge_danger`

| option | expert (n=203) | AR (n=2 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash_attack | 38.4 | 0.0 | 0.0 | 0.0 |
| shield_on | 33.5 | 0.0 | 0.0 | 0.0 |
| special | 11.8 | 0.0 | 0.0 | 0.0 |
| smash | 7.9 | 0.0 | 0.0 | 0.0 |
| dash | 4.4 | 100.0 | 0.0 | 0.0 |
| grab | 3.4 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.5 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 29.4 | 20.7 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `offstage`

| option | expert (n=12343) | AR (n=3 ⚠) | IND (n=7 ⚠) | ep10_cpu (n=37) |
|---|---:|---:|---:|---:|
| special | 57.1 | 33.3 | 28.6 | 32.4 |
| aerial | 21.1 | 0.0 | 0.0 | 10.8 |
| airdodge | 7.1 | 0.0 | 0.0 | 16.2 |
| ledge_getup | 3.4 | 0.0 | 14.3 | 0.0 |
| ledge_jump | 2.8 | 0.0 | 0.0 | 0.0 |
| dash | 1.8 | 66.7 | 42.9 | 0.0 |
| ledge_roll | 1.6 | 0.0 | 0.0 | 0.0 |
| double_jump | 1.4 | 0.0 | 14.3 | 2.7 |
| shield_on | 1.3 | 0.0 | 0.0 | 8.1 |
| ledge_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.3 | 0.0 | 0.0 | 0.0 |
| tilt | 0.1 | 0.0 | 0.0 | 2.7 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.1 | 0.0 | 0.0 | 2.7 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 40.8 | 3.0 | 6.8 | 37.0 |
| **TV vs expert** | – | – | – | 0.46 |
| **KL(set‖expert)** | – | – | – | 1.06 |

### `ledge_hang`

| option | expert (n=1085) | AR (n=0 ⚠) | IND (n=1 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| ledge_getup | 38.4 | 0.0 | 100.0 | 0.0 |
| ledge_jump | 31.7 | 0.0 | 0.0 | 0.0 |
| ledge_roll | 18.5 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 10.7 | 0.0 | 0.0 | 0.0 |
| special | 0.5 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 45.8 | 0.0 | 300.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `respawn_invincible`

| option | expert (n=8665) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=17 ⚠) |
|---|---:|---:|---:|---:|
| dash | 56.6 | 0.0 | 0.0 | 0.0 |
| double_jump | 12.3 | 0.0 | 0.0 | 0.0 |
| dashdance | 9.3 | 0.0 | 0.0 | 0.0 |
| aerial | 5.8 | 0.0 | 0.0 | 0.0 |
| special | 5.5 | 0.0 | 0.0 | 17.6 |
| wavedash | 4.0 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 0.0 | 0.0 | 17.6 |
| waveland | 2.2 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.4 | 0.0 | 0.0 | 11.8 |
| dash_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| grab | 0.1 | 0.0 | 0.0 | 23.5 |
| jab | 0.1 | 0.0 | 0.0 | 5.9 |
| roll_backward | 0.1 | 0.0 | 0.0 | 17.6 |
| spotdodge | 0.0 | 0.0 | 0.0 | 5.9 |
| ledge_jump | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 142.5 | 0.0 | 0.0 | 38.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `post_kill_neutral`

| option | expert (n=29324) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=42) |
|---|---:|---:|---:|---:|
| dash | 45.8 | 0.0 | 0.0 | 4.8 |
| double_jump | 14.0 | 0.0 | 0.0 | 4.8 |
| special | 7.5 | 0.0 | 0.0 | 11.9 |
| aerial | 6.7 | 0.0 | 0.0 | 0.0 |
| dashdance | 6.5 | 0.0 | 0.0 | 0.0 |
| shield_on | 4.6 | 0.0 | 0.0 | 14.3 |
| airdodge | 4.2 | 0.0 | 0.0 | 0.0 |
| waveland | 4.1 | 0.0 | 0.0 | 0.0 |
| wavedash | 3.3 | 0.0 | 0.0 | 2.4 |
| roll_forward | 0.5 | 0.0 | 0.0 | 4.8 |
| grab | 0.4 | 0.0 | 0.0 | 31.0 |
| roll_backward | 0.4 | 0.0 | 0.0 | 11.9 |
| missed_tech | 0.4 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.3 | 0.0 | 0.0 | 4.8 |
| ledge_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 205.8 | 0.0 | 0.0 | 72.0 |
| **TV vs expert** | – | – | – | 0.74 |
| **KL(set‖expert)** | – | – | – | 2.10 |

### `percent_lead`

| option | expert (n=27351) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=215) |
|---|---:|---:|---:|---:|
| dash | 34.1 | 0.0 | 0.0 | 0.5 |
| double_jump | 17.5 | 0.0 | 0.0 | 5.1 |
| aerial | 14.5 | 0.0 | 0.0 | 6.0 |
| special | 8.7 | 0.0 | 0.0 | 18.6 |
| shield_on | 5.0 | 0.0 | 0.0 | 7.4 |
| dashdance | 2.8 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.5 | 0.0 | 0.0 | 6.5 |
| tilt | 2.0 | 0.0 | 0.0 | 3.3 |
| airdodge | 1.8 | 0.0 | 0.0 | 0.5 |
| waveland | 1.7 | 0.0 | 0.0 | 0.5 |
| grab | 1.7 | 0.0 | 0.0 | 20.0 |
| smash | 1.6 | 0.0 | 0.0 | 4.7 |
| jab | 1.1 | 0.0 | 0.0 | 3.3 |
| throw | 0.9 | 0.0 | 0.0 | 3.7 |
| dash_attack | 0.7 | 0.0 | 0.0 | 0.5 |
| *options / min in situation* | 138.8 | 0.0 | 0.0 | 69.9 |
| **TV vs expert** | – | – | – | 0.61 |
| **KL(set‖expert)** | – | – | – | 1.19 |

### `percent_deficit`

| option | expert (n=18268) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=8 ⚠) |
|---|---:|---:|---:|---:|
| dash | 27.3 | 0.0 | 0.0 | 0.0 |
| double_jump | 13.0 | 0.0 | 0.0 | 0.0 |
| aerial | 12.9 | 0.0 | 0.0 | 0.0 |
| special | 12.8 | 0.0 | 0.0 | 12.5 |
| shield_on | 8.5 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.9 | 0.0 | 0.0 | 12.5 |
| dashdance | 2.7 | 0.0 | 0.0 | 0.0 |
| waveland | 2.7 | 0.0 | 0.0 | 0.0 |
| grab | 2.6 | 0.0 | 0.0 | 25.0 |
| missed_tech | 2.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.0 | 0.0 | 0.0 | 0.0 |
| tilt | 1.4 | 0.0 | 0.0 | 0.0 |
| throw | 1.3 | 0.0 | 0.0 | 12.5 |
| smash | 1.1 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 1.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 97.0 | 0.0 | 0.0 | 47.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

