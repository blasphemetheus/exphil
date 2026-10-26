# Mamba resync schedules — AI-generated measurements

Each row is a separate ten-second phase at 60 Hz. One late batch affects eight bots.
Round two reverses schedule order. Setup and carry validation are outside the timed phases.

| Client | Mode | Schedule | Round | Load | Response p95 / p99 ms | Missed / offered calls |
|---|---|---|---:|---|---:|---:|
| host | actors | aligned | 1 | idle | 68.20 / 81.74 | 935 / 4800 |
| host | actors | aligned | 1 | matmul | 69.47 / 85.96 | 976 / 4800 |
| host | batch | aligned | 1 | idle | 15.70 / 32.99 | 29 / 600 |
| host | batch | aligned | 1 | matmul | 14.39 / 31.16 | 26 / 600 |
| host | actors | spread | 1 | idle | 16.11 / 18.47 | 139 / 4800 |
| host | actors | spread | 1 | matmul | 16.47 / 19.39 | 197 / 4800 |
| host | batch | spread | 1 | idle | 14.28 / 15.02 | 1 / 600 |
| host | batch | spread | 1 | matmul | 14.28 / 15.56 | 0 / 600 |
| host | actors | spread | 2 | idle | 16.02 / 18.04 | 129 / 4800 |
| host | actors | spread | 2 | matmul | 16.45 / 18.25 | 170 / 4800 |
| host | batch | spread | 2 | idle | 14.46 / 15.69 | 0 / 600 |
| host | batch | spread | 2 | matmul | 14.14 / 15.06 | 0 / 600 |
| host | actors | aligned | 2 | idle | 68.60 / 81.09 | 912 / 4800 |
| host | actors | aligned | 2 | matmul | 70.18 / 82.73 | 928 / 4800 |
| host | batch | aligned | 2 | idle | 13.51 / 31.39 | 24 / 600 |
| host | batch | aligned | 2 | matmul | 14.65 / 31.67 | 25 / 600 |
| cuda | actors | aligned | 1 | idle | 30.34 / 41.11 | 462 / 4800 |
| cuda | actors | aligned | 1 | matmul | 38.99 / 45.99 | 844 / 4800 |
| cuda | batch | aligned | 1 | idle | 3.61 / 8.63 | 0 / 600 |
| cuda | batch | aligned | 1 | matmul | 4.22 / 9.98 | 0 / 600 |
| cuda | actors | spread | 1 | idle | 12.62 / 14.02 | 0 / 4800 |
| cuda | actors | spread | 1 | matmul | 16.67 / 18.17 | 243 / 4800 |
| cuda | batch | spread | 1 | idle | 7.58 / 8.08 | 0 / 600 |
| cuda | batch | spread | 1 | matmul | 8.98 / 9.56 | 0 / 600 |
| cuda | actors | spread | 2 | idle | 12.62 / 13.85 | 0 / 4800 |
| cuda | actors | spread | 2 | matmul | 16.84 / 18.22 | 299 / 4800 |
| cuda | batch | spread | 2 | idle | 7.19 / 7.74 | 0 / 600 |
| cuda | batch | spread | 2 | matmul | 8.90 / 9.36 | 0 / 600 |
| cuda | actors | aligned | 2 | idle | 30.69 / 41.32 | 464 / 4800 |
| cuda | actors | aligned | 2 | matmul | 39.76 / 46.25 | 829 / 4800 |
| cuda | batch | aligned | 2 | idle | 3.65 / 8.41 | 0 / 600 |
| cuda | batch | aligned | 2 | matmul | 4.23 / 10.14 | 0 / 600 |
