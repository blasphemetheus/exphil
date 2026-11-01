# Complete controller latency — AI-generated

Checkpoint: `checkpoints/coh_fox_mamba_p314_ep4/model_best_policy.bin`. Replay: `/home/blewf/Slippi/2026-10-Mainline/Game_20261022T170737.slp`. 10-second phases, 8 bots, 60 Hz.
Agent input through returned controller, including encoding, recurrent carry/replay and stochastic sampling.
Dolphin transport, emulator work, rendering and configured game-frame delay are excluded.
Warm execution only: compilation/history fill are recorded as setup, not included in percentiles.
Service measures request-to-return time; response also includes lateness relative to the scheduled frame.
One late batch affects all rows. The competing matmul is synthetic GPU load, not a training simulation.
Desktop GPU processes remain. Other compute-process observations invalidate a run.
Resync uses the benchmark's validated phase assignment, with schedule order reversed in round two.
Carry differences are recorded, not treated as policy-quality or batch/single equivalence proof.

| Client | Mode | Schedule | Round | Load | Service median ms | Service p99 ms | Response p99 ms | Response max ms | Missed/offered calls |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| cuda | single | aligned | 0 | desktop only | 2.41 | 6.35 | 7.61 | 8.68 | 0/600 |
| cuda | single | aligned | 0 | matmul | 7.43 | 12.40 | 13.51 | 14.32 | 0/600 |
| cuda | actors | aligned | 0 | desktop only | 12.23 | 41.74 | 43.42 | 46.88 | 1176/4800 |
| cuda | actors | aligned | 0 | matmul | 18.62 | 48.83 | 1854.67 | 1865.04 | 4800/4800 |
| cuda | batch | aligned | 0 | desktop only | 2.17 | 7.00 | 8.27 | 9.06 | 0/600 |
| cuda | batch | aligned | 0 | matmul | 2.96 | 8.76 | 9.94 | 10.38 | 0/600 |
| cuda | actors | aligned | 1 | desktop only | 11.99 | 42.66 | 43.62 | 47.02 | 1059/4800 |
| cuda | actors | aligned | 1 | matmul | 19.00 | 50.56 | 2168.76 | 2182.29 | 4800/4800 |
| cuda | batch | aligned | 1 | desktop only | 2.11 | 6.90 | 7.93 | 8.76 | 0/600 |
| cuda | batch | aligned | 1 | matmul | 3.02 | 8.60 | 9.90 | 10.66 | 0/600 |
| cuda | actors | spread | 1 | desktop only | 11.99 | 19.61 | 20.93 | 28.85 | 539/4800 |
| cuda | actors | spread | 1 | matmul | 19.15 | 26.54 | 1975.60 | 1996.80 | 4800/4800 |
| cuda | batch | spread | 1 | desktop only | 2.15 | 6.32 | 7.85 | 9.31 | 0/600 |
| cuda | batch | spread | 1 | matmul | 2.99 | 8.64 | 9.79 | 11.59 | 0/600 |
| cuda | actors | spread | 2 | desktop only | 12.60 | 20.72 | 25.08 | 39.36 | 1040/4800 |
| cuda | actors | spread | 2 | matmul | 19.28 | 26.61 | 2034.57 | 2056.65 | 4800/4800 |
| cuda | batch | spread | 2 | desktop only | 2.18 | 6.43 | 7.97 | 8.64 | 0/600 |
| cuda | batch | spread | 2 | matmul | 3.22 | 8.27 | 9.68 | 10.74 | 0/600 |
| cuda | actors | aligned | 2 | desktop only | 12.34 | 42.18 | 43.25 | 46.66 | 1074/4800 |
| cuda | actors | aligned | 2 | matmul | 19.05 | 49.92 | 2144.32 | 2160.67 | 4800/4800 |
| cuda | batch | aligned | 2 | desktop only | 2.12 | 6.75 | 7.96 | 8.80 | 0/600 |
| cuda | batch | aligned | 2 | matmul | 2.97 | 8.77 | 10.13 | 10.89 | 0/600 |
| host | single | aligned | 0 | desktop only | 1.37 | 11.11 | 12.19 | 13.78 | 0/600 |
| host | single | aligned | 0 | matmul | 1.31 | 10.74 | 11.83 | 13.50 | 0/600 |
| host | actors | aligned | 0 | desktop only | 6.34 | 73.23 | 74.68 | 87.25 | 984/4800 |
| host | actors | aligned | 0 | matmul | 6.52 | 76.31 | 77.09 | 87.04 | 998/4800 |
| host | batch | aligned | 0 | desktop only | 2.67 | 63.69 | 64.51 | 69.36 | 80/600 |
| host | batch | aligned | 0 | matmul | 2.53 | 69.81 | 70.73 | 72.09 | 81/600 |
| host | actors | aligned | 1 | desktop only | 6.07 | 72.52 | 73.61 | 82.34 | 892/4800 |
| host | actors | aligned | 1 | matmul | 6.21 | 77.63 | 78.72 | 88.12 | 968/4800 |
| host | batch | aligned | 1 | desktop only | 2.65 | 67.79 | 68.87 | 71.19 | 76/600 |
| host | batch | aligned | 1 | matmul | 2.56 | 67.23 | 67.92 | 72.89 | 76/600 |
| host | actors | spread | 1 | desktop only | 6.25 | 18.92 | 19.85 | 24.60 | 277/4800 |
| host | actors | spread | 1 | matmul | 6.44 | 19.38 | 20.45 | 23.77 | 543/4800 |
| host | batch | spread | 1 | desktop only | 2.84 | 15.50 | 17.01 | 18.35 | 9/600 |
| host | batch | spread | 1 | matmul | 2.66 | 14.62 | 15.82 | 16.55 | 0/600 |
| host | actors | spread | 2 | desktop only | 6.51 | 19.00 | 20.21 | 26.50 | 452/4800 |
| host | actors | spread | 2 | matmul | 6.95 | 20.03 | 21.16 | 23.88 | 798/4800 |
| host | batch | spread | 2 | desktop only | 2.94 | 16.05 | 17.09 | 18.37 | 11/600 |
| host | batch | spread | 2 | matmul | 2.65 | 14.78 | 16.13 | 18.63 | 4/600 |
| host | actors | aligned | 2 | desktop only | 6.30 | 73.39 | 74.20 | 81.27 | 920/4800 |
| host | actors | aligned | 2 | matmul | 6.31 | 81.67 | 82.61 | 90.14 | 1019/4800 |
| host | batch | aligned | 2 | desktop only | 2.68 | 64.81 | 65.44 | 69.94 | 76/600 |
| host | batch | aligned | 2 | matmul | 2.54 | 68.36 | 69.78 | 73.15 | 76/600 |

**AI-generated poem**

A frame arrives, a choice returns;  
We count the time while memory turns.
