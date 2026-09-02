# Situation → next-option histograms (A1) and distribution distance (B2)

Sets: expert: 600 files, 226534 events · AR: 24 files, 3712 events · IND: 24 files, 3160 events · ep10_cpu: 8 files, 1249 events.
Subject port: expert 1, others 1. Situation = labels active on the
frame BEFORE the option fired (`Situations`); option = `Options.events`. Shares are
% of that set's options in that situation. TV = total-variation distance to the
expert's histogram (0 = identical, 1 = disjoint); KL is smoothed (ε=1e-3).
Cells with n < 20 are marked ⚠ / skipped. `_any` = all frames.

## Summary — TV distance to expert, per situation

| situation | expert n | AR TV (n) | IND TV (n) | ep10_cpu TV (n) |
|---|---:|---:|---:|---:|
| _any | 226534 | 0.62 (3712) | 0.61 (3160) | 0.55 (1249) |
| neutral | 157942 | 0.65 (2793) | 0.63 (2422) | 0.57 (858) |
| approach | 61170 | 0.65 (798) | 0.63 (614) | 0.62 (245) |
| retreat | 37800 | 0.66 (276) | 0.65 (163) | 0.59 (74) |
| advantage | 37713 | 0.55 (693) | 0.57 (542) | 0.49 (302) |
| disadvantage | 9992 | 0.46 (126) | 0.50 (121) | 0.46 (46) |
| conversion_open | 98251 | 0.53 (1592) | 0.54 (1304) | 0.47 (558) |
| combo_active | 41986 | 0.53 (699) | 0.58 (617) | 0.43 (249) |
| juggle | 8811 | 0.68 (36) | 0.80 (24) | – (16) |
| tech_chase | 6765 | 0.60 (183) | 0.60 (119) | 0.60 (69) |
| ledge_trap | 987 | – (0) | – (4) | – (4) |
| edgeguard | 28592 | 0.62 (292) | 0.66 (218) | 0.58 (62) |
| shield_pressure_ours | 5869 | – (0) | – (0) | – (0) |
| pummel_throw_decision | 3050 | 0.52 (314) | 0.50 (290) | 0.59 (89) |
| shield_pressure_theirs | 2726 | 0.13 (270) | 0.11 (248) | 0.28 (47) |
| being_edgeguarded | 6446 | 0.58 (102) | 0.47 (68) | 0.40 (36) |
| recovery_low | 7831 | 0.26 (23) | – (16) | – (11) |
| recovery_high | 4512 | 0.67 (94) | 0.52 (71) | 0.51 (26) |
| cornered | 18649 | 0.59 (363) | 0.59 (369) | 0.60 (189) |
| edge_danger | 203 | – (1) | – (0) | – (0) |
| offstage | 12343 | 0.59 (117) | 0.55 (87) | 0.46 (37) |
| ledge_hang | 1085 | – (1) | – (1) | – (0) |
| respawn_invincible | 8665 | 0.79 (47) | 0.73 (37) | – (17) |
| post_kill_neutral | 29324 | 0.74 (133) | 0.72 (102) | 0.74 (42) |
| percent_lead | 27351 | 0.65 (1019) | 0.66 (907) | 0.61 (215) |
| percent_deficit | 18268 | 0.69 (50) | 0.59 (69) | – (8) |

**Mean TV over reported situations:** AR: 0.58 over 21 situations · IND: 0.58 over 20 situations · ep10_cpu: 0.53 over 17 situations

## Per situation

### `_any`

| option | expert (n=226534) | AR (n=3712) | IND (n=3160) | ep10_cpu (n=1249) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 0.7 | 0.8 | 1.2 |
| double_jump | 15.4 | 3.8 | 4.5 | 5.8 |
| aerial | 14.0 | 4.2 | 3.6 | 5.8 |
| special | 9.9 | 14.8 | 13.8 | 21.6 |
| shield_on | 6.2 | 12.8 | 13.0 | 6.7 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.3 | 1.2 | 1.9 | 2.0 |
| wavedash | 2.2 | 7.0 | 7.4 | 6.2 |
| grab | 2.2 | 17.7 | 21.4 | 16.3 |
| waveland | 2.2 | 0.8 | 1.1 | 0.9 |
| tilt | 1.9 | 0.4 | 0.4 | 3.0 |
| smash | 1.4 | 4.2 | 3.0 | 3.3 |
| missed_tech | 1.4 | 1.4 | 1.5 | 1.8 |
| throw | 1.2 | 3.0 | 3.4 | 2.0 |
| jab | 0.9 | 5.5 | 6.3 | 3.4 |
| *options / min in situation* | 125.9 | 75.8 | 64.5 | 76.5 |
| **TV vs expert** | – | 0.62 | 0.61 | 0.55 |
| **KL(set‖expert)** | – | 1.11 | 1.09 | 0.95 |

### `neutral`

| option | expert (n=157942) | AR (n=2793) | IND (n=2422) | ep10_cpu (n=858) |
|---|---:|---:|---:|---:|
| dash | 31.6 | 0.5 | 0.4 | 1.3 |
| aerial | 15.6 | 3.6 | 3.5 | 5.1 |
| double_jump | 15.5 | 3.3 | 4.5 | 6.1 |
| special | 10.1 | 14.9 | 14.3 | 22.0 |
| shield_on | 6.9 | 14.9 | 14.5 | 7.0 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| airdodge | 2.6 | 0.9 | 1.8 | 1.7 |
| waveland | 2.5 | 0.8 | 1.4 | 1.0 |
| tilt | 2.2 | 0.6 | 0.4 | 3.5 |
| grab | 2.1 | 18.5 | 22.4 | 17.4 |
| wavedash | 2.0 | 7.7 | 8.6 | 7.2 |
| smash | 1.3 | 4.9 | 3.3 | 3.5 |
| jab | 1.0 | 4.7 | 5.5 | 2.3 |
| roll_forward | 0.8 | 6.2 | 5.3 | 3.8 |
| spotdodge | 0.7 | 9.7 | 9.4 | 15.3 |
| *options / min in situation* | 158.0 | 92.2 | 87.8 | 89.3 |
| **TV vs expert** | – | 0.65 | 0.63 | 0.57 |
| **KL(set‖expert)** | – | 1.20 | 1.15 | 1.03 |

### `approach`

| option | expert (n=61170) | AR (n=798) | IND (n=614) | ep10_cpu (n=245) |
|---|---:|---:|---:|---:|
| dash | 25.6 | 0.4 | 0.7 | 2.0 |
| aerial | 21.9 | 4.5 | 3.6 | 3.7 |
| double_jump | 15.9 | 4.1 | 5.4 | 3.3 |
| special | 9.7 | 12.7 | 8.1 | 17.1 |
| shield_on | 6.2 | 13.9 | 14.5 | 7.3 |
| dashdance | 3.1 | 0.0 | 0.0 | 0.0 |
| waveland | 2.9 | 1.6 | 3.4 | 1.6 |
| airdodge | 2.8 | 0.6 | 1.5 | 1.2 |
| tilt | 2.3 | 0.1 | 0.2 | 3.7 |
| grab | 2.1 | 16.9 | 20.8 | 11.8 |
| wavedash | 1.8 | 11.4 | 14.0 | 13.5 |
| smash | 1.6 | 3.9 | 2.9 | 2.9 |
| dash_attack | 0.9 | 0.0 | 0.0 | 0.0 |
| jab | 0.9 | 5.3 | 7.0 | 4.5 |
| spotdodge | 0.6 | 8.8 | 6.5 | 21.2 |
| *options / min in situation* | 168.1 | 83.5 | 76.1 | 75.8 |
| **TV vs expert** | – | 0.65 | 0.63 | 0.62 |
| **KL(set‖expert)** | – | 1.25 | 1.22 | 1.28 |

### `retreat`

| option | expert (n=37800) | AR (n=276) | IND (n=163) | ep10_cpu (n=74) |
|---|---:|---:|---:|---:|
| dash | 42.2 | 0.0 | 0.0 | 1.4 |
| double_jump | 13.0 | 3.6 | 1.2 | 5.4 |
| special | 10.4 | 20.7 | 14.7 | 33.8 |
| aerial | 9.9 | 3.6 | 4.9 | 10.8 |
| shield_on | 6.4 | 19.2 | 15.3 | 6.8 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| airdodge | 3.3 | 3.3 | 8.0 | 5.4 |
| waveland | 2.6 | 0.4 | 1.2 | 0.0 |
| wavedash | 2.2 | 3.3 | 3.1 | 0.0 |
| tilt | 1.3 | 0.0 | 1.2 | 0.0 |
| grab | 1.1 | 15.9 | 26.4 | 8.1 |
| roll_forward | 0.8 | 5.4 | 6.1 | 6.8 |
| smash | 0.7 | 5.1 | 2.5 | 2.7 |
| jab | 0.6 | 3.6 | 5.5 | 1.4 |
| spotdodge | 0.6 | 5.8 | 5.5 | 10.8 |
| *options / min in situation* | 144.8 | 72.0 | 56.4 | 73.3 |
| **TV vs expert** | – | 0.66 | 0.65 | 0.59 |
| **KL(set‖expert)** | – | 1.31 | 1.43 | 1.10 |

### `advantage`

| option | expert (n=37713) | AR (n=693) | IND (n=542) | ep10_cpu (n=302) |
|---|---:|---:|---:|---:|
| dash | 31.3 | 1.6 | 2.4 | 1.3 |
| double_jump | 20.1 | 6.8 | 5.9 | 7.0 |
| aerial | 12.7 | 6.3 | 4.2 | 9.3 |
| special | 8.1 | 14.4 | 13.1 | 23.2 |
| throw | 7.0 | 16.0 | 19.6 | 8.3 |
| grab | 4.1 | 17.2 | 19.0 | 14.6 |
| wavedash | 3.3 | 5.8 | 4.6 | 5.3 |
| smash | 2.9 | 1.3 | 2.0 | 3.3 |
| shield_on | 2.9 | 6.6 | 8.1 | 5.0 |
| tilt | 2.3 | 0.0 | 0.2 | 2.3 |
| airdodge | 1.3 | 1.2 | 0.9 | 1.7 |
| waveland | 1.2 | 0.7 | 0.2 | 0.7 |
| jab | 0.8 | 9.1 | 10.3 | 6.6 |
| dash_attack | 0.8 | 0.0 | 0.0 | 0.0 |
| dashdance | 0.6 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 137.1 | 58.4 | 37.0 | 65.6 |
| **TV vs expert** | – | 0.55 | 0.57 | 0.49 |
| **KL(set‖expert)** | – | 0.91 | 0.92 | 0.83 |

### `disadvantage`

| option | expert (n=9992) | AR (n=126) | IND (n=121) | ep10_cpu (n=46) |
|---|---:|---:|---:|---:|
| missed_tech | 24.0 | 25.4 | 25.6 | 21.7 |
| shield_on | 16.1 | 7.9 | 8.3 | 10.9 |
| special | 15.6 | 12.7 | 9.9 | 17.4 |
| tech_roll | 13.2 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 13.1 | 0.0 | 0.0 | 0.0 |
| aerial | 5.0 | 3.2 | 1.7 | 0.0 |
| getup_stand | 4.9 | 0.0 | 0.0 | 0.0 |
| getup_attack | 3.9 | 34.1 | 33.1 | 39.1 |
| dash | 1.7 | 0.0 | 0.8 | 0.0 |
| spotdodge | 1.1 | 2.4 | 0.8 | 2.2 |
| airdodge | 0.4 | 2.4 | 5.0 | 4.3 |
| jab | 0.3 | 1.6 | 0.0 | 2.2 |
| tilt | 0.3 | 0.0 | 0.8 | 0.0 |
| smash | 0.2 | 4.8 | 3.3 | 2.2 |
| grab | 0.2 | 5.6 | 10.7 | 0.0 |
| *options / min in situation* | 36.1 | 74.6 | 70.1 | 81.3 |
| **TV vs expert** | – | 0.46 | 0.50 | 0.46 |
| **KL(set‖expert)** | – | 1.00 | 1.16 | 1.01 |

### `conversion_open`

| option | expert (n=98251) | AR (n=1592) | IND (n=1304) | ep10_cpu (n=558) |
|---|---:|---:|---:|---:|
| dash | 25.7 | 0.9 | 0.7 | 1.6 |
| double_jump | 15.9 | 4.6 | 4.4 | 6.8 |
| aerial | 13.5 | 4.6 | 3.8 | 7.0 |
| special | 10.9 | 16.0 | 16.3 | 24.6 |
| shield_on | 6.7 | 10.9 | 12.2 | 5.2 |
| grab | 3.1 | 17.3 | 21.4 | 14.0 |
| missed_tech | 3.0 | 3.3 | 3.6 | 4.1 |
| throw | 2.6 | 5.2 | 4.9 | 2.7 |
| wavedash | 2.4 | 4.8 | 4.1 | 4.3 |
| tilt | 2.2 | 0.8 | 0.5 | 3.2 |
| smash | 2.0 | 3.6 | 2.9 | 2.2 |
| airdodge | 1.6 | 1.3 | 1.7 | 2.2 |
| waveland | 1.4 | 0.8 | 0.7 | 0.7 |
| tech_in_place | 1.4 | 0.0 | 0.0 | 0.0 |
| tech_roll | 1.3 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 101.0 | 84.9 | 74.0 | 83.7 |
| **TV vs expert** | – | 0.53 | 0.54 | 0.47 |
| **KL(set‖expert)** | – | 0.81 | 0.84 | 0.66 |

### `combo_active`

| option | expert (n=41986) | AR (n=699) | IND (n=617) | ep10_cpu (n=249) |
|---|---:|---:|---:|---:|
| dash | 29.5 | 1.9 | 1.6 | 2.0 |
| double_jump | 21.0 | 7.6 | 6.8 | 8.8 |
| aerial | 13.2 | 6.3 | 4.2 | 12.9 |
| special | 9.8 | 14.7 | 11.8 | 21.7 |
| throw | 6.3 | 15.9 | 17.2 | 10.0 |
| wavedash | 3.3 | 3.4 | 2.9 | 4.0 |
| shield_on | 3.3 | 6.7 | 8.6 | 4.0 |
| grab | 2.8 | 17.9 | 23.5 | 13.3 |
| smash | 2.5 | 1.9 | 1.1 | 3.6 |
| tilt | 2.5 | 0.1 | 0.2 | 3.2 |
| airdodge | 1.5 | 1.0 | 1.0 | 0.8 |
| waveland | 1.2 | 0.9 | 0.3 | 0.4 |
| jab | 0.9 | 7.3 | 9.9 | 4.4 |
| dashdance | 0.7 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.5 | 0.0 | 0.0 | 0.4 |
| *options / min in situation* | 129.3 | 48.7 | 34.3 | 58.0 |
| **TV vs expert** | – | 0.53 | 0.58 | 0.43 |
| **KL(set‖expert)** | – | 0.90 | 1.03 | 0.64 |

### `juggle`

| option | expert (n=8811) | AR (n=36) | IND (n=24) | ep10_cpu (n=16 ⚠) |
|---|---:|---:|---:|---:|
| dash | 33.7 | 0.0 | 0.0 | 0.0 |
| double_jump | 31.6 | 11.1 | 8.3 | 12.5 |
| aerial | 20.3 | 11.1 | 8.3 | 18.8 |
| special | 3.3 | 13.9 | 16.7 | 12.5 |
| smash | 2.2 | 2.8 | 0.0 | 18.8 |
| tilt | 1.9 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.9 | 2.8 | 0.0 | 6.3 |
| waveland | 1.3 | 0.0 | 0.0 | 0.0 |
| wavedash | 1.2 | 2.8 | 0.0 | 0.0 |
| airdodge | 0.9 | 0.0 | 0.0 | 6.3 |
| dashdance | 0.4 | 0.0 | 0.0 | 0.0 |
| grab | 0.3 | 27.8 | 33.3 | 12.5 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.2 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 2.8 | 0.0 | 6.3 |
| *options / min in situation* | 122.7 | 62.5 | 39.6 | 72.7 |
| **TV vs expert** | – | 0.68 | 0.80 | – |
| **KL(set‖expert)** | – | 2.25 | 3.04 | – |

### `tech_chase`

| option | expert (n=6765) | AR (n=183) | IND (n=119) | ep10_cpu (n=69) |
|---|---:|---:|---:|---:|
| dash | 39.5 | 1.6 | 1.7 | 0.0 |
| double_jump | 23.3 | 13.1 | 10.9 | 13.0 |
| grab | 12.0 | 23.5 | 28.6 | 23.2 |
| smash | 6.4 | 2.7 | 5.0 | 2.9 |
| tilt | 4.0 | 0.0 | 0.0 | 2.9 |
| shield_on | 3.4 | 8.7 | 9.2 | 5.8 |
| dash_attack | 2.9 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 20.8 | 18.5 | 23.2 |
| jab | 2.6 | 6.6 | 6.7 | 1.4 |
| aerial | 1.5 | 4.9 | 3.4 | 7.2 |
| dashdance | 1.1 | 0.0 | 0.0 | 0.0 |
| throw | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 3.8 | 7.6 | 15.9 |
| roll_backward | 0.1 | 3.8 | 7.6 | 0.0 |
| roll_forward | 0.0 | 9.8 | 0.8 | 4.3 |
| *options / min in situation* | 175.9 | 138.3 | 121.2 | 126.9 |
| **TV vs expert** | – | 0.60 | 0.60 | 0.60 |
| **KL(set‖expert)** | – | 1.24 | 1.16 | 1.47 |

### `ledge_trap`

| option | expert (n=987) | AR (n=0 ⚠) | IND (n=4 ⚠) | ep10_cpu (n=4 ⚠) |
|---|---:|---:|---:|---:|
| dash | 55.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 21.5 | 0.0 | 0.0 | 50.0 |
| shield_on | 8.1 | 0.0 | 0.0 | 25.0 |
| dashdance | 5.2 | 0.0 | 0.0 | 0.0 |
| special | 2.9 | 0.0 | 25.0 | 0.0 |
| tilt | 2.2 | 0.0 | 0.0 | 0.0 |
| jab | 1.4 | 0.0 | 25.0 | 0.0 |
| smash | 1.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.5 | 0.0 | 25.0 | 0.0 |
| aerial | 0.3 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 0.0 | 25.0 | 25.0 |
| roll_backward | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 172.0 | 0.0 | 113.4 | 464.5 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `edgeguard`

| option | expert (n=28592) | AR (n=292) | IND (n=218) | ep10_cpu (n=62) |
|---|---:|---:|---:|---:|
| dash | 38.4 | 2.4 | 2.8 | 4.8 |
| double_jump | 21.4 | 7.5 | 4.6 | 8.1 |
| special | 9.3 | 15.1 | 11.0 | 9.7 |
| aerial | 9.1 | 5.5 | 3.2 | 6.5 |
| wavedash | 4.3 | 6.8 | 8.7 | 4.8 |
| shield_on | 4.2 | 7.5 | 7.8 | 6.5 |
| dashdance | 3.4 | 0.0 | 0.0 | 0.0 |
| waveland | 2.1 | 1.0 | 0.9 | 0.0 |
| airdodge | 2.1 | 0.7 | 2.3 | 0.0 |
| tilt | 1.4 | 0.0 | 0.0 | 3.2 |
| smash | 1.2 | 2.4 | 1.4 | 1.6 |
| jab | 0.7 | 10.3 | 12.4 | 12.9 |
| roll_forward | 0.4 | 7.9 | 8.7 | 6.5 |
| roll_backward | 0.4 | 6.2 | 2.8 | 4.8 |
| dash_attack | 0.4 | 0.0 | 0.0 | 1.6 |
| *options / min in situation* | 154.0 | 85.6 | 65.7 | 74.6 |
| **TV vs expert** | – | 0.62 | 0.66 | 0.58 |
| **KL(set‖expert)** | – | 1.57 | 1.97 | 1.65 |

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

| option | expert (n=3050) | AR (n=314) | IND (n=290) | ep10_cpu (n=89) |
|---|---:|---:|---:|---:|
| throw | 85.9 | 35.4 | 36.6 | 28.1 |
| shield_on | 9.9 | 29.9 | 28.6 | 21.3 |
| dash | 1.7 | 0.0 | 1.0 | 1.1 |
| grab | 0.9 | 1.6 | 3.8 | 0.0 |
| spotdodge | 0.6 | 6.4 | 3.4 | 22.5 |
| tilt | 0.4 | 0.6 | 0.0 | 3.4 |
| jab | 0.2 | 9.9 | 10.3 | 7.9 |
| special | 0.2 | 8.3 | 11.4 | 11.2 |
| smash | 0.2 | 8.0 | 4.8 | 4.5 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 87.8 | 40.9 | 23.1 | 29.1 |
| **TV vs expert** | – | 0.52 | 0.50 | 0.59 |
| **KL(set‖expert)** | – | 1.06 | 1.02 | 1.49 |

### `shield_pressure_theirs`

| option | expert (n=2726) | AR (n=270) | IND (n=248) | ep10_cpu (n=47) |
|---|---:|---:|---:|---:|
| grab | 29.0 | 19.3 | 25.8 | 27.7 |
| roll_forward | 24.9 | 25.9 | 19.8 | 10.6 |
| spotdodge | 23.3 | 30.0 | 34.3 | 48.9 |
| roll_backward | 19.8 | 24.8 | 19.8 | 10.6 |
| shield_on | 2.2 | 0.0 | 0.0 | 0.0 |
| dash | 0.3 | 0.0 | 0.0 | 0.0 |
| tech_roll | 0.1 | 0.0 | 0.0 | 0.0 |
| jab | 0.1 | 0.0 | 0.4 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| tech_in_place | 0.0 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.0 | 0.0 | 0.0 | 0.0 |
| dash_attack | 0.0 | 0.0 | 0.0 | 0.0 |
| tilt | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 89.4 | 1601.3 | 1226.4 | 829.4 |
| **TV vs expert** | – | 0.13 | 0.11 | 0.28 |
| **KL(set‖expert)** | – | 0.05 | 0.05 | 0.25 |

### `being_edgeguarded`

| option | expert (n=6446) | AR (n=102) | IND (n=68) | ep10_cpu (n=36) |
|---|---:|---:|---:|---:|
| special | 62.1 | 19.6 | 23.5 | 33.3 |
| aerial | 18.8 | 6.9 | 11.8 | 11.1 |
| airdodge | 11.5 | 11.8 | 19.1 | 16.7 |
| dash | 2.2 | 0.0 | 2.9 | 0.0 |
| shield_on | 1.8 | 13.7 | 7.4 | 8.3 |
| double_jump | 1.4 | 3.9 | 8.8 | 2.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.2 | 6.9 | 1.5 | 2.8 |
| tilt | 0.2 | 0.0 | 0.0 | 2.8 |
| waveland | 0.2 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.2 | 2.9 | 4.4 | 2.8 |
| tech_in_place | 0.1 | 0.0 | 0.0 | 0.0 |
| getup_stand | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.1 | 11.8 | 2.9 | 2.8 |
| grab | 0.1 | 15.7 | 13.2 | 11.1 |
| *options / min in situation* | 36.0 | 40.7 | 29.5 | 36.8 |
| **TV vs expert** | – | 0.58 | 0.47 | 0.40 |
| **KL(set‖expert)** | – | 1.64 | 0.97 | 0.80 |

### `recovery_low`

| option | expert (n=7831) | AR (n=23) | IND (n=16 ⚠) | ep10_cpu (n=11 ⚠) |
|---|---:|---:|---:|---:|
| special | 60.7 | 47.8 | 37.5 | 54.5 |
| aerial | 18.4 | 17.4 | 12.5 | 9.1 |
| ledge_getup | 5.3 | 0.0 | 0.0 | 0.0 |
| airdodge | 4.5 | 30.4 | 43.8 | 36.4 |
| ledge_jump | 4.4 | 4.3 | 0.0 | 0.0 |
| ledge_roll | 2.6 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 1.5 | 0.0 | 6.3 | 0.0 |
| dash | 0.8 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.8 | 0.0 | 0.0 | 0.0 |
| shield_on | 0.4 | 0.0 | 0.0 | 0.0 |
| missed_tech | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.1 | 0.0 | 0.0 | 0.0 |
| smash | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_backward | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 50.2 | 15.2 | 12.1 | 16.5 |
| **TV vs expert** | – | 0.26 | – | – |
| **KL(set‖expert)** | – | 0.42 | – | – |

### `recovery_high`

| option | expert (n=4512) | AR (n=94) | IND (n=71) | ep10_cpu (n=26) |
|---|---:|---:|---:|---:|
| special | 50.9 | 16.0 | 18.3 | 23.1 |
| aerial | 25.8 | 5.3 | 9.9 | 11.5 |
| airdodge | 11.5 | 5.3 | 9.9 | 7.7 |
| dash | 3.5 | 0.0 | 4.2 | 0.0 |
| shield_on | 2.8 | 16.0 | 11.3 | 11.5 |
| double_jump | 2.6 | 4.3 | 8.5 | 3.8 |
| missed_tech | 0.5 | 0.0 | 0.0 | 0.0 |
| tilt | 0.4 | 0.0 | 0.0 | 3.8 |
| roll_forward | 0.3 | 7.4 | 4.2 | 3.8 |
| jab | 0.2 | 4.3 | 2.8 | 3.8 |
| smash | 0.2 | 1.1 | 0.0 | 0.0 |
| getup_stand | 0.2 | 0.0 | 0.0 | 0.0 |
| grab | 0.2 | 21.3 | 19.7 | 19.2 |
| tech_in_place | 0.2 | 0.0 | 0.0 | 0.0 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 30.8 | 67.9 | 53.5 | 77.7 |
| **TV vs expert** | – | 0.67 | 0.52 | 0.51 |
| **KL(set‖expert)** | – | 1.93 | 1.29 | 1.28 |

### `cornered`

| option | expert (n=18649) | AR (n=363) | IND (n=369) | ep10_cpu (n=189) |
|---|---:|---:|---:|---:|
| dash | 37.9 | 0.6 | 0.5 | 1.6 |
| double_jump | 19.3 | 5.5 | 4.9 | 9.0 |
| shield_on | 15.8 | 14.9 | 17.6 | 7.9 |
| grab | 4.1 | 21.2 | 25.5 | 17.5 |
| special | 3.8 | 11.8 | 13.3 | 21.2 |
| dashdance | 2.9 | 0.0 | 0.0 | 0.0 |
| tilt | 2.8 | 1.1 | 0.5 | 4.8 |
| roll_forward | 2.7 | 9.9 | 6.8 | 4.2 |
| smash | 1.8 | 5.5 | 2.7 | 4.2 |
| throw | 1.7 | 5.2 | 5.7 | 2.6 |
| spotdodge | 1.5 | 9.1 | 9.5 | 18.0 |
| getup_stand | 1.1 | 0.0 | 0.0 | 0.0 |
| aerial | 1.1 | 1.1 | 1.1 | 2.6 |
| roll_backward | 1.0 | 8.0 | 3.8 | 0.5 |
| jab | 0.9 | 3.0 | 5.4 | 2.6 |
| *options / min in situation* | 169.9 | 76.5 | 75.9 | 88.2 |
| **TV vs expert** | – | 0.59 | 0.59 | 0.60 |
| **KL(set‖expert)** | – | 0.99 | 1.01 | 1.03 |

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
| *options / min in situation* | 29.4 | 900.0 | 0.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `offstage`

| option | expert (n=12343) | AR (n=117) | IND (n=87) | ep10_cpu (n=37) |
|---|---:|---:|---:|---:|
| special | 57.1 | 22.2 | 21.8 | 32.4 |
| aerial | 21.1 | 7.7 | 10.3 | 10.8 |
| airdodge | 7.1 | 10.3 | 16.1 | 16.2 |
| ledge_getup | 3.4 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 2.8 | 0.9 | 0.0 | 0.0 |
| dash | 1.8 | 0.0 | 3.4 | 0.0 |
| ledge_roll | 1.6 | 0.0 | 0.0 | 0.0 |
| double_jump | 1.4 | 3.4 | 6.9 | 2.7 |
| shield_on | 1.3 | 12.8 | 9.2 | 8.1 |
| ledge_attack | 0.9 | 0.0 | 1.1 | 0.0 |
| missed_tech | 0.3 | 0.0 | 0.0 | 0.0 |
| tilt | 0.1 | 0.0 | 0.0 | 2.7 |
| waveland | 0.1 | 0.0 | 0.0 | 0.0 |
| roll_forward | 0.1 | 6.0 | 3.4 | 2.7 |
| smash | 0.1 | 0.9 | 0.0 | 0.0 |
| *options / min in situation* | 40.8 | 40.4 | 32.8 | 37.0 |
| **TV vs expert** | – | 0.59 | 0.55 | 0.46 |
| **KL(set‖expert)** | – | 1.71 | 1.30 | 1.06 |

### `ledge_hang`

| option | expert (n=1085) | AR (n=1 ⚠) | IND (n=1 ⚠) | ep10_cpu (n=0 ⚠) |
|---|---:|---:|---:|---:|
| ledge_getup | 38.4 | 0.0 | 0.0 | 0.0 |
| ledge_jump | 31.7 | 100.0 | 0.0 | 0.0 |
| ledge_roll | 18.5 | 0.0 | 0.0 | 0.0 |
| ledge_attack | 10.7 | 0.0 | 100.0 | 0.0 |
| special | 0.5 | 0.0 | 0.0 | 0.0 |
| double_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 45.8 | 163.6 | 400.0 | 0.0 |
| **TV vs expert** | – | – | – | – |
| **KL(set‖expert)** | – | – | – | – |

### `respawn_invincible`

| option | expert (n=8665) | AR (n=47) | IND (n=37) | ep10_cpu (n=17 ⚠) |
|---|---:|---:|---:|---:|
| dash | 56.6 | 0.0 | 2.7 | 0.0 |
| double_jump | 12.3 | 2.1 | 5.4 | 0.0 |
| dashdance | 9.3 | 0.0 | 0.0 | 0.0 |
| aerial | 5.8 | 10.6 | 10.8 | 0.0 |
| special | 5.5 | 17.0 | 13.5 | 17.6 |
| wavedash | 4.0 | 8.5 | 5.4 | 0.0 |
| airdodge | 2.3 | 17.0 | 13.5 | 17.6 |
| waveland | 2.2 | 0.0 | 0.0 | 0.0 |
| shield_on | 1.4 | 4.3 | 8.1 | 11.8 |
| dash_attack | 0.1 | 0.0 | 0.0 | 0.0 |
| grab | 0.1 | 12.8 | 18.9 | 23.5 |
| jab | 0.1 | 10.6 | 8.1 | 5.9 |
| roll_backward | 0.1 | 2.1 | 5.4 | 17.6 |
| spotdodge | 0.0 | 6.4 | 5.4 | 5.9 |
| ledge_jump | 0.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 142.5 | 36.2 | 33.6 | 38.5 |
| **TV vs expert** | – | 0.79 | 0.73 | – |
| **KL(set‖expert)** | – | 2.22 | 2.10 | – |

### `post_kill_neutral`

| option | expert (n=29324) | AR (n=133) | IND (n=102) | ep10_cpu (n=42) |
|---|---:|---:|---:|---:|
| dash | 45.8 | 0.0 | 1.0 | 4.8 |
| double_jump | 14.0 | 5.3 | 2.9 | 4.8 |
| special | 7.5 | 15.8 | 16.7 | 11.9 |
| aerial | 6.7 | 3.0 | 6.9 | 0.0 |
| dashdance | 6.5 | 0.0 | 0.0 | 0.0 |
| shield_on | 4.6 | 12.8 | 13.7 | 14.3 |
| airdodge | 4.2 | 0.8 | 1.0 | 0.0 |
| waveland | 4.1 | 0.0 | 0.0 | 0.0 |
| wavedash | 3.3 | 7.5 | 2.9 | 2.4 |
| roll_forward | 0.5 | 5.3 | 0.0 | 4.8 |
| grab | 0.4 | 20.3 | 31.4 | 31.0 |
| roll_backward | 0.4 | 7.5 | 2.9 | 11.9 |
| missed_tech | 0.4 | 0.0 | 0.0 | 0.0 |
| spotdodge | 0.3 | 7.5 | 10.8 | 4.8 |
| ledge_jump | 0.2 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 205.8 | 60.4 | 61.2 | 72.0 |
| **TV vs expert** | – | 0.74 | 0.72 | 0.74 |
| **KL(set‖expert)** | – | 1.94 | 2.10 | 2.10 |

### `percent_lead`

| option | expert (n=27351) | AR (n=1019) | IND (n=907) | ep10_cpu (n=215) |
|---|---:|---:|---:|---:|
| dash | 34.1 | 0.7 | 0.7 | 0.5 |
| double_jump | 17.5 | 4.3 | 4.9 | 5.1 |
| aerial | 14.5 | 4.4 | 3.7 | 6.0 |
| special | 8.7 | 14.0 | 13.3 | 18.6 |
| shield_on | 5.0 | 11.9 | 12.7 | 7.4 |
| dashdance | 2.8 | 0.0 | 0.0 | 0.0 |
| wavedash | 2.5 | 6.2 | 7.9 | 6.5 |
| tilt | 2.0 | 0.4 | 0.2 | 3.3 |
| airdodge | 1.8 | 1.0 | 1.1 | 0.5 |
| waveland | 1.7 | 0.6 | 0.2 | 0.5 |
| grab | 1.7 | 17.9 | 21.2 | 20.0 |
| smash | 1.6 | 4.3 | 3.2 | 4.7 |
| jab | 1.1 | 6.8 | 6.0 | 3.3 |
| throw | 0.9 | 3.0 | 3.4 | 3.7 |
| dash_attack | 0.7 | 0.0 | 0.0 | 0.5 |
| *options / min in situation* | 138.8 | 72.0 | 60.7 | 69.9 |
| **TV vs expert** | – | 0.65 | 0.66 | 0.61 |
| **KL(set‖expert)** | – | 1.26 | 1.25 | 1.19 |

### `percent_deficit`

| option | expert (n=18268) | AR (n=50) | IND (n=69) | ep10_cpu (n=8 ⚠) |
|---|---:|---:|---:|---:|
| dash | 27.3 | 0.0 | 0.0 | 0.0 |
| double_jump | 13.0 | 2.0 | 5.8 | 0.0 |
| aerial | 12.9 | 0.0 | 0.0 | 0.0 |
| special | 12.8 | 6.0 | 20.3 | 12.5 |
| shield_on | 8.5 | 16.0 | 11.6 | 0.0 |
| airdodge | 2.9 | 2.0 | 4.3 | 12.5 |
| dashdance | 2.7 | 0.0 | 0.0 | 0.0 |
| waveland | 2.7 | 0.0 | 1.4 | 0.0 |
| grab | 2.6 | 24.0 | 21.7 | 25.0 |
| missed_tech | 2.3 | 2.0 | 0.0 | 0.0 |
| wavedash | 2.0 | 8.0 | 8.7 | 0.0 |
| tilt | 1.4 | 0.0 | 0.0 | 0.0 |
| throw | 1.3 | 2.0 | 1.4 | 12.5 |
| smash | 1.1 | 6.0 | 4.3 | 0.0 |
| tech_in_place | 1.0 | 0.0 | 0.0 | 0.0 |
| *options / min in situation* | 97.0 | 70.4 | 91.6 | 47.0 |
| **TV vs expert** | – | 0.69 | 0.59 | – |
| **KL(set‖expert)** | – | 1.41 | 1.11 | – |

