# Punishable-commitment scorecard (F2)

Sets: expert: 166 files · bot: 30 files.
Committal = smash, dash_attack, grab, spotdodge, roll_forward, roll_backward (Options events).
In-threat = opponent not in hitstun, |dx| < 30, |dy| < 20 at the
event frame. Punished = subject enters hitstun within 60 f. Cells under
n=10 report "–".

| commitment | expert (n=3036) | bot (n=526) |
|---|---:|---:|
| committal options / min | 6.53 | 7.39 |
| IN-THREAT committals / min | 4.35 | 5.50 |
| in-threat share of committals % | 66.7 | 74.3 |
| P(punished | in-threat committal) % | 22.9 (2026) | 31.7 (391) |
| P(punished | out-of-threat committal) % | 15.5 (1010) | 19.3 (135) |
| in-threat smash: P(punished) % | 21.8 (486) | 46.0 (63) |
| in-threat dash_attack: P(punished) % | 19.8 (197) | 27.9 (43) |
| in-threat grab: P(punished) % | 19.8 (763) | 31.1 (167) |
| in-threat spotdodge: P(punished) % | 37.1 (240) | 28.2 (39) |
| in-threat roll_forward: P(punished) % | 26.0 (208) | 26.8 (41) |
| in-threat roll_backward: P(punished) % | 18.2 (132) | 23.7 (38) |

Read: the expert commits too — the difference that matters is
P(punished | in-threat), i.e. whether commitments are TIMED (opponent
committed/landing) or thrown out raw. High in-threat share + high
P(punished) = "puts itself into whiff punish" (C1's mechanism).
