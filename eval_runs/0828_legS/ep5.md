# Leg S — selection headroom (pass@k)

Policy: `fox_gen_v1_20260825_210355_ep5.bin`
Replays: 20 files, port 1
Frames: decision only —
45007 candidates from 199255 frames
(22.6%),
2000 scored at n=16 (one forward per frame, 60 frames of same-game history), temperature 0.5,
stick tolerance 0.0625.

| head | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | **headroom** |
|---|---|---|---|---|---|---|
| joint | 13.3 | 19.8 | 26.9 | 34.2 | 41.0 | **+27.7** |
| buttons | 41.3 | 58.5 | 74.0 | 86.0 | 93.5 | **+52.2** |
| main | 36.0 | 42.7 | 48.5 | 54.1 | 59.5 | **+23.5** |
| c | 91.3 | 92.3 | 92.9 | 93.4 | 93.8 | **+2.5** |
| shoulder | 85.9 | 89.7 | 92.2 | 94.1 | 95.7 | **+9.7** |

**Headroom** = pass@16 - pass@1: what a perfect selector could
recover without changing a weight.

## Verdict

SELECTION headroom is large (+27.7 pts). The right action is in the distribution and the decode is not picking it. A value model has real room; this is the upper bound on what it could recover.

## Limits

pass@k assumes a perfect selector and is therefore an UPPER BOUND on what a
value model can buy. It is open-loop and cannot see closed-loop drift. A
master's exact action is not the only correct play, so absolute pass@1 is
not a skill score — only differences and the gap are meaningful.
