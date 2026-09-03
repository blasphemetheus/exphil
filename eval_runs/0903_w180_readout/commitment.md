# Punishable-commitment scorecard (F2)

Sets: w180: 4 files · v14: 4 files.
Committal = smash, dash_attack, grab, spotdodge, roll_forward, roll_backward (Options events).
In-threat = opponent not in hitstun, |dx| < 30, |dy| < 20 at the
event frame. Punished = subject enters hitstun within 60 f. Cells under
n=10 report "–".

| commitment | w180 (n=271) | v14 (n=225) |
|---|---:|---:|
| committal options / min | 33.41 | 32.55 |
| IN-THREAT committals / min | 23.18 | 20.40 |
| in-threat share of committals % | 69.4 | 62.7 |
| P(punished | in-threat committal) % | 52.7 (188) | 44.7 (141) |
| P(punished | out-of-threat committal) % | 12.0 (83) | 3.6 (84) |
| in-threat smash: P(punished) % | 83.3 (12) | 90.9 (11) |
| in-threat dash_attack: P(punished) % | – (0) | – (1) |
| in-threat grab: P(punished) % | 59.2 (98) | 59.0 (39) |
| in-threat spotdodge: P(punished) % | 55.6 (54) | 51.9 (52) |
| in-threat roll_forward: P(punished) % | 6.7 (15) | 10.0 (20) |
| in-threat roll_backward: P(punished) % | – (9) | 5.6 (18) |

Read: the expert commits too — the difference that matters is
P(punished | in-threat), i.e. whether commitments are TIMED (opponent
committed/landing) or thrown out raw. High in-threat share + high
P(punished) = "puts itself into whiff punish" (C1's mechanism).
