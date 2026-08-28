# Leg S — selection headroom (pass@k)

Policy: `fox_gen_v1_20260825_210355_ep10.bin`
Replays: 20 files, port 1
Frames: decision only —
45007 candidates from 199255 frames
(22.6%),
2000 scored at n=16, temperature 0.5,
stick tolerance 0.0625.

| head | pass@1 | pass@2 | pass@4 | pass@8 | pass@16 | **headroom** |
|---|---|---|---|---|---|---|
| joint | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | **+0.0** |
| buttons | 8.0 | 8.0 | 8.0 | 8.0 | 8.0 | **+0.0** |
| main | 10.9 | 10.9 | 10.9 | 10.9 | 10.9 | **+0.0** |
| c | 89.6 | 89.6 | 89.6 | 89.6 | 89.6 | **+0.0** |
| shoulder | 76.5 | 76.5 | 76.5 | 76.5 | 76.5 | **+0.0** |

**Headroom** = pass@16 - pass@1: what a perfect selector could
recover without changing a weight.

## Verdict

The right action is NOT in the distribution even at k=16. No selector can help. Pay for Leg C (capacity ladder) and Leg D (data ladder) to find out which.

## Limits

pass@k assumes a perfect selector and is therefore an UPPER BOUND on what a
value model can buy. It is open-loop and cannot see closed-loop drift. A
master's exact action is not the only correct play, so absolute pass@1 is
not a skill score — only differences and the gap are meaningful.
