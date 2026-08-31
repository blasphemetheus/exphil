# Punishable-commitment scorecard (F2)

Sets: expert: 166 files · AR_human: 13 files · IND_human: 6 files · B1_human: 7 files.
Committal = smash, dash_attack, grab, spotdodge, roll_forward, roll_backward (Options events).
In-threat = opponent not in hitstun, |dx| < 30, |dy| < 20 at the
event frame. Punished = subject enters hitstun within 60 f. Cells under
n=10 report "–".

| commitment | expert (n=3072) | AR_human (n=411) | IND_human (n=184) | B1_human (n=143) |
|---|---:|---:|---:|---:|
| committal options / min | 6.22 | 19.76 | 22.14 | 20.06 |
| IN-THREAT committals / min | 4.13 | 10.86 | 13.00 | 12.63 |
| in-threat share of committals % | 66.3 | 55.0 | 58.7 | 62.9 |
| P(punished | in-threat committal) % | 64.9 (2037) | 54.4 (226) | 47.2 (108) | 56.7 (90) |
| P(punished | out-of-threat committal) % | 53.5 (1035) | 36.8 (185) | 31.6 (76) | 47.2 (53) |
| in-threat smash: P(punished) % | 84.6 (481) | 83.3 (30) | – (8) | – (3) |
| in-threat dash_attack: P(punished) % | 88.4 (198) | – (1) | – (0) | – (1) |
| in-threat grab: P(punished) % | 52.9 (773) | 54.8 (62) | 58.5 (41) | 76.9 (39) |
| in-threat spotdodge: P(punished) % | 69.3 (244) | 52.8 (72) | 34.5 (29) | 50.0 (22) |
| in-threat roll_forward: P(punished) % | 50.5 (208) | 45.5 (33) | 45.0 (20) | 21.4 (14) |
| in-threat roll_backward: P(punished) % | 42.9 (133) | 35.7 (28) | 20.0 (10) | 45.5 (11) |

Read: the expert commits too — the difference that matters is
P(punished | in-threat), i.e. whether commitments are TIMED (opponent
committed/landing) or thrown out raw. High in-threat share + high
P(punished) = "puts itself into whiff punish" (C1's mechanism).
