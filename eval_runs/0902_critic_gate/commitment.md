# Punishable-commitment scorecard (F2)

Sets: expert: 600 files · CRITIC: 8 files · BASE: 8 files.
Committal = smash, dash_attack, grab, spotdodge, roll_forward, roll_backward (Options events).
In-threat = opponent not in hitstun, |dx| < 30, |dy| < 20 at the
event frame. Punished = subject enters hitstun within 60 f. Cells under
n=10 report "–".

| commitment | expert (n=13229) | CRITIC (n=107) | BASE (n=519) |
|---|---:|---:|---:|
| committal options / min | 7.35 | 9.50 | 32.69 |
| IN-THREAT committals / min | 5.05 | 5.86 | 20.59 |
| in-threat share of committals % | 68.7 | 61.7 | 63.0 |
| P(punished | in-threat committal) % | 66.6 (9084) | 78.8 (66) | 52.3 (327) |
| P(punished | out-of-threat committal) % | 52.6 (4145) | 26.8 (41) | 17.7 (192) |
| in-threat smash: P(punished) % | 86.3 (1868) | 91.1 (56) | 76.2 (42) |
| in-threat dash_attack: P(punished) % | 85.8 (893) | – (0) | – (0) |
| in-threat grab: P(punished) % | 60.7 (3866) | – (2) | 61.5 (135) |
| in-threat spotdodge: P(punished) % | 72.1 (940) | – (4) | 59.5 (79) |
| in-threat roll_forward: P(punished) % | 44.2 (867) | – (2) | 18.2 (33) |
| in-threat roll_backward: P(punished) % | 41.1 (650) | – (2) | 7.9 (38) |

Read: the expert commits too — the difference that matters is
P(punished | in-threat), i.e. whether commitments are TIMED (opponent
committed/landing) or thrown out raw. High in-threat share + high
P(punished) = "puts itself into whiff punish" (C1's mechanism).
