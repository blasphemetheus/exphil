# Leg S — selection headroom (pass@k)

Policy: `fox_gen_v1_20260825_210355_ep10.bin`
Replays: 20 files, port 1
Frames: decision only —
18819 candidates from 168750 frames
(11.2%),
2000 scored at n=16 (one forward per frame, 60 frames of same-game history), temperature 0.5,
stick tolerance 0.0625.

| head | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | **headroom** |
|---|---|---|---|---|---|---|
| joint | 6.9 | 10.5 | 14.7 | 19.3 | 24.3 | **+17.4** |
| buttons | 35.9 | 53.4 | 69.6 | 81.3 | 89.0 | **+53.0** |
| main | 20.2 | 24.9 | 29.3 | 33.8 | 38.2 | **+18.0** |
| c | 91.9 | 93.3 | 94.0 | 94.5 | 95.0 | **+3.0** |
| shoulder | 89.3 | 91.9 | 93.6 | 94.9 | 96.0 | **+6.8** |

**Headroom** = pass@16 - pass@1: what a perfect selector could
recover without changing a weight.

## Verdict

SELECTION headroom is large (+17.4 pts). The right action is in the distribution and the decode is not picking it. A value model has real room; this is the upper bound on what it could recover.

## Limits

pass@k assumes a perfect selector and is therefore an UPPER BOUND on what a
value model can buy. It is open-loop and cannot see closed-loop drift. A
master's exact action is not the only correct play, so absolute pass@1 is
not a skill score — only differences and the gap are meaningful.
