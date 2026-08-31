# Situation → next-option histograms (A1) and distribution distance (B2)

Sets: expert: 691 files, 286239 events · AR: 8 files, 1270 events · IND: 8 files, 1072 events · ep10_cpu: 8 files, 1249 events.
Subject port: expert 1, others 1. Situation = labels active on the
frame BEFORE the option fired (`Situations`); option = `Options.events`. Shares are
% of that set's options in that situation. TV = total-variation distance to the
expert's histogram (0 = identical, 1 = disjoint); KL is smoothed (ε=1e-3).
Cells with n < 20 are marked ⚠ / skipped. `_any` = all frames.

## Summary — TV distance to expert, per situation

| situation | expert n | AR TV (n) | IND TV (n) | ep10_cpu TV (n) |
|---|---:|---:|---:|---:|
| _any | 286239 | 0.66 (1270) | 0.64 (1072) | 0.58 (1249) |
| neutral | 200029 | 0.68 (964) | 0.65 (812) | 0.60 (858) |
| approach | 77469 | 0.66 (308) | 0.65 (255) | 0.65 (245) |
| retreat | 47254 | 0.71 (81) | 0.70 (60) | 0.65 (74) |
| advantage | 42642 | 0.65 (256) | 0.64 (192) | 0.52 (302) |
| disadvantage | 12444 | 0.56 (24) | 0.55 (42) | 0.46 (46) |
| conversion_open | 116080 | 0.61 (506) | 0.60 (436) | 0.49 (558) |
| combo_active | 51073 | 0.66 (199) | 0.66 (163) | 0.46 (249) |
| juggle | 12828 | – (7) | – (13) | – (16) |
| tech_chase | 4944 | 0.71 (93) | 0.74 (55) | 0.62 (69) |
| ledge_trap | 1446 | – (0) | – (0) | – (4) |
| edgeguard | 38554 | 0.73 (39) | 0.63 (46) | 0.59 (62) |
| shield_pressure_ours | 7873 | – (0) | – (0) | – (0) |
| pummel_throw_decision | 2747 | 0.55 (150) | 0.50 (131) | 0.61 (89) |
| shield_pressure_theirs | 2584 | 0.20 (91) | 0.13 (65) | 0.28 (47) |
| being_edgeguarded | 6660 | 0.62 (106) | 0.63 (48) | 0.39 (36) |
| recovery_low | 8732 | – (14) | – (7) | – (11) |
| recovery_high | 4582 | 0.68 (97) | 0.63 (42) | 0.49 (26) |
| cornered | 21781 | 0.63 (194) | 0.63 (169) | 0.63 (189) |
| edge_danger | 211 | – (4) | – (1) | – (0) |
| offstage | 13314 | 0.62 (111) | 0.64 (49) | 0.45 (37) |
| ledge_hang | 1297 | – (1) | – (0) | – (0) |
| respawn_invincible | 11600 | – (10) | – (16) | – (17) |
| post_kill_neutral | 37682 | – (16) | 0.84 (24) | 0.75 (42) |
| percent_lead | 35453 | 0.72 (134) | 0.74 (45) | 0.64 (215) |
| percent_deficit | 21174 | 0.75 (22) | – (11) | – (8) |

**Mean TV over reported situations:** AR: 0.63 over 17 situations · IND: 0.62 over 17 situations · ep10_cpu: 0.55 over 17 situations

## Per situation

### `_any`

| option | expert (n=286239) | AR (n=1270) | IND (n=1072) | ep10_cpu (n=1249) |
|---|---:|---:|---:|---:|
| dash | 34.8 | 1.9 | 2.2 | 1.2 |
| double_jump | 16.3 | 1.7 | 2.1 | 5.8 |
| aerial | 12.3 | 2.8 | 4.0 | 5.8 |
| special | 9.7 | 12.3 | 12.5 | 21.6 |
| shield_on | 5.5 | 13.7 | 13.3 | 6.7 |
| dashdance | 3.8 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.0 | 1.7 | 2.3 | 2.0 |
| wavedash | 2.0 | 4.3 | 7.3 | 6.2 |
| waveland | 2.0 | 1.4 | 1.1 | 0.9 |
| tilt | 1.7 | 0.3 | 0.2 | 3.0 |
| grab | 1.5 | 20.6 | 23.0 | 16.3 |
| missed_tech | 1.4 | 0.6 | 1.0 | 1.8 |
| smash | 1.3 | 4.4 | 3.3 | 3.3 |
| throw | 0.8 | 3.8 | 4.7 | 2.0 |
| tech_roll | 0.8 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 136.7 | 90.7 | 73.4 | 76.5 |
| **TV vs expert** | – | 0.66 | 0.64 | 0.58 |
| **KL(set‖expert)** | – | 1.41 | 1.26 | 1.07 |

### `neutral`

| option | expert (n=200029) | AR (n=964) | IND (n=812) | ep10_cpu (n=858) |
|---|---:|---:|---:|---:|
| dash | 35.6 | 2.0 | 2.1 | 1.3 |
| double_jump | 16.4 | 1.6 | 2.2 | 6.1 |
| aerial | 13.4 | 2.4 | 4.3 | 5.1 |
| special | 9.8 | 12.7 | 12.4 | 22.0 |
| shield_on | 6.0 | 14.7 | 14.5 | 7.0 |
| dashdance | 4.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 1.2 | 2.1 | 1.7 |
| waveland | 2.3 | 1.7 | 1.5 | 1.0 |
| tilt | 1.9 | 0.3 | 0.1 | 3.5 |
| wavedash | 1.8 | 4.7 | 8.4 | 7.2 |
| grab | 1.5 | 22.4 | 25.2 | 17.4 |
| smash | 1.2 | 4.9 | 3.1 | 3.5 |
| jab | 0.8 | 3.3 | 4.3 | 2.3 |
| spotdodge | 0.7 | 12.3 | 7.4 | 15.3 |
| roll_forward | 0.6 | 6.3 | 4.9 | 3.8 |
| *options / min in situation* | 170.4 | 101.7 | 84.1 | 89.3 |
| **TV vs expert** | – | 0.68 | 0.65 | 0.60 |
| **KL(set‖expert)** | – | 1.46 | 1.32 | 1.15 |

### `approach`

| option | expert (n=77469) | AR (n=308) | IND (n=255) | ep10_cpu (n=245) |
|---|---:|---:|---:|---:|
| dash | 28.3 | 2.3 | 3.9 | 2.0 |
| aerial | 20.7 | 2.9 | 3.9 | 3.7 |
| double_jump | 17.4 | 2.9 | 2.7 | 3.3 |
| special | 9.3 | 10.4 | 9.8 | 17.1 |
| shield_on | 5.2 | 12.7 | 11.4 | 7.3 |
| dashdance | 3.9 | 0.0 | 0.0 | 0.0 |
| waveland | 2.6 | 3.2 | 2.7 | 1.6 |
| airdodge | 2.4 | 1.0 | 1.2 | 1.2 |
| grab | 1.8 | 22.4 | 23.9 | 11.8 |
| tilt | 1.8 | 0.3 | 0.0 | 3.7 |
| smash | 1.8 | 4.9 | 1.2 | 2.9 |
| wavedash | 1.6 | 7.1 | 12.9 | 13.5 |
| dash_attack | 0.8 | 0.3 | 0.0 | 0.0 |
| jab | 0.6 | 2.6 | 5.9 | 4.5 |
| spotdodge | 0.5 | 10.1 | 5.5 | 21.2 |
| *options / min in situation* | 182.6 | 98.6 | 93.2 | 75.8 |
| **TV vs expert** | – | 0.66 | 0.65 | 0.65 |
| **KL(set‖expert)** | – | 1.43 | 1.40 | 1.41 |

### `retreat`

| option | expert (n=47254) | AR (n=81) | IND (n=60) | ep10_cpu (n=74) |
|---|---:|---:|---:|---:|
| dash | 47.7 | 1.2 | 0.0 | 1.4 |
| double_jump | 12.9 | 1.2 | 1.7 | 5.4 |
| special | 10.1 | 14.8 | 8.3 | 33.8 |
| aerial | 6.9 | 1.2 | 8.3 | 10.8 |
| shield_on | 5.4 | 16.0 | 13.3 | 6.8 |
| dashdance | 4.7 | 0.0 | 0.0 | 0.0 |
| airdodge | 3.3 | 3.7 | 8.3 | 5.4 |
| waveland | 2.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 1.9 | 8.6 | 1.7 | 0.0 |
| tilt | 1.2 | 1.2 | 0.0 | 0.0 |
| grab | 0.6 | 25.9 | 31.7 | 8.1 |
| roll_forward | 0.5 | 1.2 | 3.3 | 6.8 |
| jab | 0.5 | 3.7 | 8.3 | 1.4 |
| spotdodge | 0.5 | 7.4 | 1.7 | 10.8 |
| roll_backward | 0.5 | 9.9 | 6.7 | 6.8 |
| *options / min in situation* | 155.1 | 61.1 | 49.0 | 73.3 |
| **TV vs expert** | – | 0.71 | 0.70 | 0.65 |
| **KL(set‖expert)** | – | 1.79 | 1.93 | 1.27 |

### `advantage`

| option | expert (n=42642) | AR (n=256) | IND (n=192) | ep10_cpu (n=302) |
|---|---:|---:|---:|---:|
| dash | 31.5 | 1.6 | 3.6 | 1.3 |
| double_jump | 22.4 | 2.3 | 2.1 | 7.0 |
| aerial | 13.6 | 2.7 | 3.1 | 9.3 |
| special | 7.7 | 10.9 | 9.9 | 23.2 |
| throw | 5.6 | 18.8 | 26.0 | 8.3 |
| wavedash | 3.4 | 2.7 | 5.2 | 5.3 |
| smash | 3.0 | 3.1 | 3.6 | 3.3 |
| shield_on | 2.9 | 10.9 | 9.9 | 5.0 |
| grab | 2.9 | 16.4 | 16.1 | 14.6 |
| tilt | 2.0 | 0.4 | 0.5 | 2.3 |
| airdodge | 1.2 | 1.2 | 0.0 | 1.7 |
| waveland | 1.0 | 0.8 | 0.0 | 0.7 |
| dashdance | 0.7 | 0.0 | 0.0 | 0.0 |
| jab | 0.5 | 3.9 | 4.2 | 6.6 |
| dash_attack | 0.5 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 146.9 | 91.8 | 67.3 | 65.6 |
| **TV vs expert** | – | 0.65 | 0.64 | 0.52 |
| **KL(set‖expert)** | – | 1.35 | 1.13 | 0.90 |

### `disadvantage`

| option | expert (n=12444) | AR (n=24) | IND (n=42) | ep10_cpu (n=46) |
|---|---:|---:|---:|---:|
| missed_tech | 23.9 | 16.7 | 14.3 | 21.7 |
| tech_roll | 16.4 | 0.0 | 0.0 | 0.0 |
| special | 15.4 | 12.5 | 16.7 | 17.4 |
| tech_in_place | 15.0 | 0.0 | 0.0 | 0.0 |
| shield_on | 14.8 | 8.3 | 9.5 | 10.9 |
| getup_stand | 5.2 | 0.0 | 0.0 | 0.0 |
| getup_attack | 4.2 | 25.0 | 16.7 | 39.1 |
| dash | 2.2 | 0.0 | 0.0 | 0.0 |
| aerial | 1.2 | 12.5 | 0.0 | 0.0 |
| spotdodge | 0.9 | 8.3 | 4.8 | 2.2 |
| tilt | 0.2 | 0.0 | 0.0 | 0.0 |
| jab | 0.2 | 0.0 | 2.4 | 2.2 |
| smash | 0.1 | 4.2 | 4.8 | 2.2 |
| airdodge | 0.1 | 8.3 | 9.5 | 4.3 |
| grab | 0.0 | 4.2 | 21.4 | 0.0 |
| *options / min in situation* | 36.5 | 78.8 | 98.2 | 81.3 |
| **TV vs expert** | – | 0.56 | 0.55 | 0.46 |
| **KL(set‖expert)** | – | 1.30 | 1.79 | 1.04 |

### `conversion_open`

| option | expert (n=116080) | AR (n=506) | IND (n=436) | ep10_cpu (n=558) |
|---|---:|---:|---:|---:|
| dash | 27.4 | 1.4 | 1.4 | 1.6 |
| double_jump | 17.2 | 1.4 | 1.1 | 6.8 |
| aerial | 12.4 | 3.0 | 2.8 | 7.0 |
| special | 11.1 | 11.5 | 14.0 | 24.6 |
| shield_on | 5.9 | 12.3 | 11.7 | 5.2 |
| missed_tech | 3.2 | 1.4 | 2.5 | 4.1 |
| wavedash | 2.3 | 3.4 | 4.1 | 4.3 |
| grab | 2.3 | 18.8 | 22.2 | 14.0 |
| tilt | 2.1 | 0.6 | 0.5 | 3.2 |
| throw | 2.0 | 9.1 | 9.9 | 2.7 |
| smash | 1.9 | 5.3 | 4.1 | 2.2 |
| tech_roll | 1.7 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 1.7 | 0.0 | 0.0 | 0.0 |
| dashdance | 1.7 | 0.0 | 0.0 | 0.0 |
| airdodge | 1.4 | 1.4 | 2.3 | 2.2 |
| *options / min in situation* | 105.8 | 101.6 | 86.7 | 83.7 |
| **TV vs expert** | – | 0.61 | 0.60 | 0.49 |
| **KL(set‖expert)** | – | 1.17 | 1.08 | 0.74 |

### `combo_active`

| option | expert (n=51073) | AR (n=199) | IND (n=163) | ep10_cpu (n=249) |
|---|---:|---:|---:|---:|
| dash | 30.3 | 2.5 | 4.3 | 2.0 |
| double_jump | 22.6 | 2.0 | 1.2 | 8.8 |
| aerial | 13.8 | 2.5 | 3.1 | 12.9 |
| special | 9.4 | 11.1 | 8.0 | 21.7 |
| throw | 4.7 | 24.1 | 30.7 | 10.0 |
| wavedash | 3.5 | 2.0 | 3.1 | 4.0 |
| shield_on | 3.1 | 10.6 | 9.2 | 4.0 |
| smash | 2.4 | 3.5 | 4.3 | 3.6 |
| tilt | 2.3 | 0.5 | 0.0 | 3.2 |
| grab | 2.3 | 13.1 | 18.4 | 13.3 |
| airdodge | 1.3 | 1.0 | 0.0 | 0.8 |
| waveland | 1.1 | 0.0 | 0.0 | 0.4 |
| dashdance | 0.9 | 0.0 | 0.0 | 0.0 |
| jab | 0.6 | 5.0 | 3.7 | 4.4 |
| dash_attack | 0.5 | 0.0 | 0.0 | 0.4 |
| *options / min in situation* | 140.2 | 68.7 | 43.5 | 58.0 |
| **TV vs expert** | – | 0.66 | 0.66 | 0.46 |
| **KL(set‖expert)** | – | 1.36 | 1.27 | 0.71 |

### `juggle`

| option | expert (n=12828) | AR (n=7 ⚠) | IND (n=13 ⚠) | ep10_cpu (n=16 ⚠) |
|---|---:|---:|---:|---:|
| dash | 35.1 | 0.0 | 7.7 | 0.0 |
| double_jump | 30.9 | 0.0 | 0.0 | 12.5 |
| aerial | 23.0 | 0.0 | 0.0 | 18.8 |
| special | 2.2 | 14.3 | 15.4 | 12.5 |
| shield_on | 1.9 | 0.0 | 23.1 | 6.3 |
| smash | 1.8 | 0.0 | 7.7 | 18.8 |
| tilt | 1.2 | 0.0 | 0.0 | 0.0 |
| airdodge | 0.9 | 0.0 | 0.0 | 6.3 |
| waveland | 0.8 | 0.0 | 0.0 | 0.0 |
| wavedash | 0.6 | 14.3 | 0.0 | 0.0 |
| dashdance | 0.5 | 0.0 | 0.0 | 0.0 |
| grab | 0.2 | 28.6 | 7.7 | 12.5 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.2 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.1 | 28.6 | 0.0 | 6.3 |
| *options / min in situation* | 134.3 | 37.3 | 70.9 | 72.7 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `tech_chase`

| option | expert (n=4944) | AR (n=93) | IND (n=55) | ep10_cpu (n=69) |
|---|---:|---:|---:|---:|
| dash | 41.0 | 0.0 | 0.0 | 0.0 |
| double_jump | 25.8 | 3.2 | 1.8 | 13.0 |
| grab | 9.1 | 25.8 | 30.9 | 23.2 |
| smash | 6.8 | 6.5 | 3.6 | 2.9 |
| tilt | 4.0 | 0.0 | 1.8 | 2.9 |
| shield_on | 3.4 | 16.1 | 18.2 | 5.8 |
| special | 2.5 | 9.7 | 12.7 | 23.2 |
| aerial | 2.0 | 5.4 | 3.6 | 7.2 |
| jab | 1.7 | 4.3 | 3.6 | 1.4 |
| dash_attack | 1.6 | 0.0 | 0.0 | 0.0 |
| dashdance | 1.5 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.1 | 8.6 | 10.9 | 0.0 |
| roll_forward | 0.1 | 14.0 | 9.1 | 4.3 |
| getup_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 183.1 | 140.4 | 124.6 | 126.9 |
| **TV vs expert** | – | 0.71 | 0.74 | 0.62 |
| **KL(set‖expert)** | – | 1.79 | 1.70 | 1.60 |

### `ledge_trap`

| option | expert (n=1446) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=4 ⚠) |
|---|---:|---:|---:|---:|
| dash | 55.3 | 0.0 | 0.0 | 0.0 |
| double_jump | 24.5 | 0.0 | 0.0 | 50.0 |
| shield_on | 7.7 | 0.0 | 0.0 | 25.0 |
| dashdance | 5.7 | 0.0 | 0.0 | 0.0 |
| special | 3.4 | 0.0 | 0.0 | 0.0 |
| tilt | 0.8 | 0.0 | 0.0 | 0.0 |
| jab | 0.7 | 0.0 | 0.0 | 0.0 |
| smash | 0.6 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.4 | 0.0 | 0.0 | 0.0 |
| grab | 0.3 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 0.0 | 0.0 | 25.0 |
| roll_backward | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 191.2 | 0.0 | 0.0 | 464.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `edgeguard`

| option | expert (n=38554) | AR (n=39) | IND (n=46) | ep10_cpu (n=62) |
|---|---:|---:|---:|---:|
| dash | 40.4 | 2.6 | 8.7 | 4.8 |
| double_jump | 21.0 | 2.6 | 0.0 | 8.1 |
| special | 8.7 | 20.5 | 23.9 | 9.7 |
| aerial | 8.1 | 2.6 | 8.7 | 6.5 |
| dashdance | 4.6 | 0.0 | 0.0 | 0.0 |
| shield_on | 4.5 | 7.7 | 4.3 | 6.5 |
| wavedash | 3.8 | 7.7 | 13.0 | 4.8 |
| waveland | 1.9 | 0.0 | 0.0 | 0.0 |
| airdodge | 1.8 | 0.0 | 0.0 | 0.0 |
| smash | 1.1 | 0.0 | 2.2 | 1.6 |
| tilt | 1.0 | 0.0 | 0.0 | 3.2 |
| roll_backward | 0.6 | 5.1 | 4.3 | 4.8 |
| roll_forward | 0.5 | 12.8 | 8.7 | 6.5 |
| jab | 0.4 | 10.3 | 6.5 | 12.9 |
| missed_tech | 0.4 | 2.6 | 0.0 | 0.0 |
| *options / min in situation* | 170.2 | 70.7 | 79.4 | 74.6 |
| **TV vs expert** | – | 0.73 | 0.63 | 0.59 |
| **KL(set‖expert)** | – | 1.83 | 1.40 | 1.69 |

### `shield_pressure_ours`

| option | expert (n=7873) | AR (n=0 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| aerial | 21.4 | 0.0 | 0.0 | 0.0 |
| double_jump | 18.8 | 0.0 | 0.0 | 0.0 |
| dash | 16.3 | 0.0 | 0.0 | 0.0 |
| special | 15.2 | 0.0 | 0.0 | 0.0 |
| shield_on | 5.4 | 0.0 | 0.0 | 0.0 |
| grab | 5.4 | 0.0 | 0.0 | 0.0 |
| tilt | 4.7 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.4 | 0.0 | 0.0 | 0.0 |
| jab | 2.4 | 0.0 | 0.0 | 0.0 |
| smash | 1.7 | 0.0 | 0.0 | 0.0 |
| waveland | 1.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 1.0 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.8 | 0.0 | 0.0 | 0.0 |
| dashdance | 0.8 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 210.7 | 0.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `pummel_throw_decision`

| option | expert (n=2747) | AR (n=150) | IND (n=131) | ep10_cpu (n=89) |
|---|---:|---:|---:|---:|
| throw | 87.4 | 32.0 | 38.2 | 28.1 |
| shield_on | 8.2 | 36.7 | 32.8 | 21.3 |
| dash | 1.9 | 4.0 | 3.1 | 1.1 |
| spotdodge | 0.7 | 8.0 | 2.3 | 22.5 |
| grab | 0.6 | 2.0 | 0.0 | 0.0 |
| special | 0.4 | 6.7 | 13.7 | 11.2 |
| tilt | 0.4 | 0.7 | 0.0 | 3.4 |
| jab | 0.3 | 2.7 | 6.9 | 7.9 |
| smash | 0.1 | 7.3 | 3.1 | 4.5 |
| *options / min in situation* | 98.1 | 67.0 | 57.4 | 29.1 |
| **TV vs expert** | – | 0.55 | 0.50 | 0.61 |
| **KL(set‖expert)** | – | 0.97 | 0.91 | 1.42 |

### `shield_pressure_theirs`

| option | expert (n=2584) | AR (n=91) | IND (n=65) | ep10_cpu (n=47) |
|---|---:|---:|---:|---:|
| spotdodge | 28.8 | 31.9 | 41.5 | 48.9 |
| roll_forward | 25.5 | 24.2 | 21.5 | 10.6 |
| grab | 21.3 | 6.6 | 15.4 | 27.7 |
| roll_backward | 20.9 | 37.4 | 21.5 | 10.6 |
| shield_on | 2.4 | 0.0 | 0.0 | 0.0 |
| dash | 0.3 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 0.2 | 0.0 | 0.0 | 0.0 |
| smash | 0.2 | 0.0 | 0.0 | 0.0 |
| special | 0.2 | 0.0 | 0.0 | 2.1 |
| tech_roll | 0.2 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.0 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 77.1 | 2213.5 | 1857.1 | 829.4 |
| **TV vs expert** | – | 0.20 | 0.13 | 0.28 |
| **KL(set‖expert)** | – | 0.15 | 0.06 | 0.20 |

### `being_edgeguarded`

| option | expert (n=6660) | AR (n=106) | IND (n=48) | ep10_cpu (n=36) |
|---|---:|---:|---:|---:|
| special | 68.3 | 19.8 | 10.4 | 33.3 |
| airdodge | 13.8 | 7.5 | 12.5 | 16.7 |
| aerial | 8.6 | 6.6 | 14.6 | 11.1 |
| dash | 2.9 | 0.0 | 2.1 | 0.0 |
| shield_on | 2.4 | 12.3 | 14.6 | 8.3 |
| double_jump | 1.6 | 0.9 | 0.0 | 2.8 |
| missed_tech | 0.6 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.3 | 14.2 | 2.1 | 2.8 |
| roll_forward | 0.2 | 1.9 | 8.3 | 2.8 |
| tilt | 0.2 | 0.0 | 0.0 | 2.8 |
| roll_backward | 0.2 | 4.7 | 2.1 | 2.8 |
| tech_roll | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 3.8 | 2.1 | 0.0 |
| grab | 0.1 | 23.6 | 18.8 | 11.1 |
| *options / min in situation* | 32.4 | 70.5 | 37.8 | 36.8 |
| **TV vs expert** | – | 0.62 | 0.63 | 0.39 |
| **KL(set‖expert)** | – | 1.93 | 1.85 | 0.79 |

### `recovery_low`

| option | expert (n=8732) | AR (n=14 ⚠) | IND (n=7 ⚠) | ep10_cpu (n=11 ⚠) |
|---|---:|---:|---:|---:|
| special | 65.3 | 35.7 | 14.3 | 54.5 |
| aerial | 12.5 | 28.6 | 57.1 | 9.1 |
| ledge_getup | 6.0 | 0.0 | 0.0 | 0.0 |
| airdodge | 4.6 | 28.6 | 28.6 | 36.4 |
| ledge_jump | 4.3 | 0.0 | 0.0 | 0.0 |
| ledge_roll | 2.9 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 1.5 | 7.1 | 0.0 | 0.0 |
| double_jump | 0.9 | 0.0 | 0.0 | 0.0 |
| dash | 0.7 | 0.0 | 0.0 | 0.0 |
| shield_on | 0.5 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.3 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 47.7 | 19.9 | 9.3 | 16.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `recovery_high`

| option | expert (n=4582) | AR (n=97) | IND (n=42) | ep10_cpu (n=26) |
|---|---:|---:|---:|---:|
| special | 55.7 | 18.6 | 9.5 | 23.1 |
| aerial | 15.3 | 3.1 | 7.1 | 11.5 |
| airdodge | 13.3 | 4.1 | 11.9 | 7.7 |
| dash | 5.1 | 0.0 | 2.4 | 0.0 |
| shield_on | 3.9 | 13.4 | 16.7 | 11.5 |
| double_jump | 3.3 | 1.0 | 0.0 | 3.8 |
| missed_tech | 0.6 | 0.0 | 0.0 | 0.0 |
| tilt | 0.4 | 0.0 | 0.0 | 3.8 |
| spotdodge | 0.3 | 16.5 | 2.4 | 3.8 |
| roll_forward | 0.3 | 2.1 | 9.5 | 3.8 |
| roll_backward | 0.3 | 5.2 | 2.4 | 3.8 |
| jab | 0.3 | 1.0 | 9.5 | 3.8 |
| smash | 0.2 | 5.2 | 2.4 | 0.0 |
| tech_roll | 0.2 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 27.9 | 109.7 | 74.3 | 77.7 |
| **TV vs expert** | – | 0.68 | 0.63 | 0.49 |
| **KL(set‖expert)** | – | 2.14 | 1.86 | 1.27 |

### `cornered`

| option | expert (n=21781) | AR (n=194) | IND (n=169) | ep10_cpu (n=189) |
|---|---:|---:|---:|---:|
| dash | 41.6 | 1.5 | 3.6 | 1.6 |
| double_jump | 19.4 | 3.6 | 1.8 | 9.0 |
| shield_on | 14.2 | 15.5 | 16.0 | 7.9 |
| special | 4.2 | 10.8 | 17.2 | 21.2 |
| dashdance | 3.8 | 0.0 | 0.0 | 0.0 |
| grab | 2.8 | 21.6 | 25.4 | 17.5 |
| tilt | 2.4 | 0.5 | 0.0 | 4.8 |
| roll_forward | 2.1 | 8.2 | 7.1 | 4.2 |
| smash | 1.8 | 6.2 | 4.1 | 4.2 |
| spotdodge | 1.4 | 15.5 | 9.5 | 18.0 |
| getup_stand | 1.1 | 0.0 | 0.6 | 0.0 |
| throw | 1.0 | 4.1 | 4.1 | 2.6 |
| aerial | 1.0 | 1.0 | 1.2 | 2.6 |
| roll_backward | 0.8 | 8.8 | 3.0 | 0.5 |
| jab | 0.7 | 1.5 | 4.7 | 2.6 |
| *options / min in situation* | 178.2 | 111.6 | 64.4 | 88.2 |
| **TV vs expert** | – | 0.63 | 0.63 | 0.63 |
| **KL(set‖expert)** | – | 1.21 | 1.13 | 1.12 |

### `edge_danger`

| option | expert (n=211) | AR (n=4 ⚠) | IND (n=1 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| dash_attack | 53.6 | 0.0 | 0.0 | 0.0 |
| shield_on | 30.3 | 0.0 | 0.0 | 0.0 |
| dash | 9.0 | 0.0 | 0.0 | 0.0 |
| smash | 5.2 | 0.0 | 0.0 | 0.0 |
| grab | 1.4 | 0.0 | 100.0 | 0.0 |
| special | 0.5 | 25.0 | 0.0 | 0.0 |
| roll_forward | 0.0 | 75.0 | 0.0 | 0.0 |
| *options / min in situation* | 24.9 | 3600.0 | 1800.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `offstage`

| option | expert (n=13314) | AR (n=111) | IND (n=49) | ep10_cpu (n=37) |
|---|---:|---:|---:|---:|
| special | 62.0 | 20.7 | 10.2 | 32.4 |
| aerial | 13.5 | 6.3 | 14.3 | 10.8 |
| airdodge | 7.6 | 7.2 | 14.3 | 16.2 |
| ledge_getup | 4.0 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 2.8 | 0.0 | 0.0 | 0.0 |
| dash | 2.2 | 0.0 | 2.0 | 0.0 |
| ledge_roll | 1.9 | 0.0 | 0.0 | 0.0 |
| double_jump | 1.7 | 0.9 | 0.0 | 2.7 |
| shield_on | 1.7 | 11.7 | 14.3 | 8.1 |
| ledge_attack | 1.0 | 0.9 | 0.0 | 0.0 |
| missed_tech | 0.4 | 0.0 | 0.0 | 0.0 |
| tilt | 0.2 | 0.0 | 0.0 | 2.7 |
| spotdodge | 0.2 | 14.4 | 2.0 | 2.7 |
| roll_backward | 0.2 | 4.5 | 2.0 | 2.7 |
| roll_forward | 0.1 | 1.8 | 8.2 | 2.7 |
| *options / min in situation* | 38.3 | 69.9 | 37.2 | 37.0 |
| **TV vs expert** | – | 0.62 | 0.64 | 0.45 |
| **KL(set‖expert)** | – | 2.05 | 1.92 | 1.03 |

### `ledge_hang`

| option | expert (n=1297) | AR (n=1 ⚠) | IND (n=0 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| ledge_getup | 40.7 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 28.7 | 0.0 | 0.0 | 0.0 |
| ledge_roll | 19.4 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 9.9 | 100.0 | 0.0 | 0.0 |
| special | 0.9 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.4 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 47.4 | 240.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `respawn_invincible`

| option | expert (n=11600) | AR (n=10 ⚠) | IND (n=16 ⚠) | ep10_cpu (n=17 ⚠) |
|---|---:|---:|---:|---:|
| dash | 57.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 11.7 | 0.0 | 0.0 | 0.0 |
| dashdance | 10.3 | 0.0 | 0.0 | 0.0 |
| special | 6.1 | 10.0 | 31.3 | 17.6 |
| aerial | 5.3 | 30.0 | 12.5 | 0.0 |
| wavedash | 3.6 | 0.0 | 0.0 | 0.0 |
| airdodge | 1.7 | 40.0 | 25.0 | 17.6 |
| waveland | 1.6 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.3 | 0.0 | 12.5 | 11.8 |
| grab | 0.1 | 0.0 | 0.0 | 23.5 |
| dash_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.0 | 0.0 | 0.0 | 17.6 |
| tilt | 0.0 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.0 | 10.0 | 6.3 | 5.9 |
| roll_forward | 0.0 | 0.0 | 6.3 | 0.0 |
| *options / min in situation* | 164.1 | 23.5 | 33.5 | 38.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `post_kill_neutral`

| option | expert (n=37682) | AR (n=16 ⚠) | IND (n=24) | ep10_cpu (n=42) |
|---|---:|---:|---:|---:|
| dash | 48.0 | 0.0 | 0.0 | 4.8 |
| double_jump | 14.3 | 0.0 | 0.0 | 4.8 |
| special | 7.5 | 12.5 | 25.0 | 11.9 |
| dashdance | 7.2 | 0.0 | 0.0 | 0.0 |
| aerial | 6.3 | 0.0 | 0.0 | 0.0 |
| shield_on | 4.0 | 12.5 | 16.7 | 14.3 |
| airdodge | 3.6 | 0.0 | 0.0 | 0.0 |
| waveland | 3.6 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.9 | 6.3 | 8.3 | 2.4 |
| roll_forward | 0.4 | 6.3 | 4.2 | 4.8 |
| grab | 0.4 | 25.0 | 25.0 | 31.0 |
| missed_tech | 0.3 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.3 | 0.0 | 8.3 | 11.9 |
| spotdodge | 0.2 | 12.5 | 4.2 | 4.8 |
| tilt | 0.2 | 0.0 | 0.0 | 2.4 |
| *options / min in situation* | 225.0 | 38.4 | 57.6 | 72.0 |
| **TV vs expert** | – | – | 0.84 | 0.75 |
| **KL(set‖expert)** | – | – | 2.26 | 2.23 |

### `percent_lead`

| option | expert (n=35453) | AR (n=134) | IND (n=45) | ep10_cpu (n=215) |
|---|---:|---:|---:|---:|
| dash | 37.3 | 3.0 | 0.0 | 0.5 |
| double_jump | 17.6 | 0.7 | 2.2 | 5.1 |
| aerial | 12.5 | 0.7 | 2.2 | 6.0 |
| special | 8.8 | 9.7 | 13.3 | 18.6 |
| shield_on | 4.2 | 12.7 | 13.3 | 7.4 |
| dashdance | 3.9 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.3 | 4.5 | 8.9 | 6.5 |
| airdodge | 1.6 | 0.7 | 0.0 | 0.5 |
| waveland | 1.6 | 0.7 | 0.0 | 0.5 |
| smash | 1.5 | 6.7 | 8.9 | 4.7 |
| tilt | 1.5 | 0.0 | 0.0 | 3.3 |
| grab | 1.4 | 26.9 | 13.3 | 20.0 |
| jab | 1.0 | 3.7 | 4.4 | 3.3 |
| throw | 0.8 | 6.7 | 4.4 | 3.7 |
| dash_attack | 0.7 | 0.0 | 0.0 | 0.5 |
| *options / min in situation* | 153.1 | 76.7 | 66.9 | 69.9 |
| **TV vs expert** | – | 0.72 | 0.74 | 0.64 |
| **KL(set‖expert)** | – | 1.65 | 1.52 | 1.28 |

### `percent_deficit`

| option | expert (n=21174) | AR (n=22) | IND (n=11 ⚠) | ep10_cpu (n=8 ⚠) |
|---|---:|---:|---:|---:|
| dash | 30.4 | 0.0 | 0.0 | 0.0 |
| double_jump | 14.2 | 0.0 | 0.0 | 0.0 |
| special | 12.7 | 13.6 | 9.1 | 12.5 |
| aerial | 11.0 | 0.0 | 0.0 | 0.0 |
| shield_on | 7.5 | 13.6 | 18.2 | 0.0 |
| dashdance | 3.6 | 0.0 | 0.0 | 0.0 |
| missed_tech | 2.6 | 0.0 | 9.1 | 0.0 |
| airdodge | 2.5 | 0.0 | 0.0 | 12.5 |
| waveland | 2.5 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.0 | 0.0 | 0.0 | 0.0 |
| grab | 1.6 | 22.7 | 9.1 | 25.0 |
| tech_roll | 1.3 | 0.0 | 0.0 | 0.0 |
| tilt | 1.3 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 1.1 | 0.0 | 0.0 | 0.0 |
| smash | 1.0 | 13.6 | 0.0 | 0.0 |
| *options / min in situation* | 102.1 | 59.8 | 84.3 | 47.0 |
| **TV vs expert** | – | 0.75 | – | – |
| **KL(set‖expert)** | – | 1.92 | – | – |

