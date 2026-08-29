# Leg S — selection headroom (pass@k)

Policy: `fox_gen_v1_20260825_210355_ep1.bin`
Replays: 20 files, port 1
Frames: decision only —
45007 candidates from 199255 frames
(22.6%),
2000 scored at n=16 (one forward per frame, 60 frames of same-game history), temperature 0.5,
stick tolerance 0.0625.

| head | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | **headroom** |
|---|---|---|---|---|---|---|
| joint | 10.4 | 15.7 | 21.7 | 28.1 | 34.3 | **+23.8** |
| buttons | 38.5 | 55.7 | 72.0 | 84.7 | 92.5 | **+54.0** |
| main | 30.2 | 36.2 | 41.4 | 46.4 | 51.9 | **+21.7** |
| c | 91.3 | 92.3 | 92.8 | 93.1 | 93.4 | **+2.1** |
| shoulder | 85.3 | 89.8 | 92.7 | 94.9 | 96.6 | **+11.3** |

**Headroom** = pass@16 - pass@1: what a perfect selector could
recover without changing a weight.

## Verdict

SELECTION headroom is large (+23.8 pts). The right action is in the distribution and the decode is not picking it. A value model has real room; this is the upper bound on what it could recover.

## Limits

pass@k assumes a perfect selector and is therefore an UPPER BOUND on what a
value model can buy. It is open-loop and cannot see closed-loop drift. A
master's exact action is not the only correct play, so absolute pass@1 is
not a skill score — only differences and the gap are meaningful.
