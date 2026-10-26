# Continuous actor throughput screen

Throughput counts individual controller actions, including every row of a batch.
Paced response latency includes delay beyond the scheduled frame time; service latency does not.
Misses finish more than 16.67 ms after scheduling (one batch miss affects all eight rows).
Short stress screens, not deployment guarantees. See RESULTS.md for methods and limitations.

| Client | Policy | Mode | GPU load | Saturated actions/s | Saturated service p95 ms | 60 Hz response p95 / p99 ms | Missed calls / offered |
|---|---|---|---|---:|---:|---:|---:|
| host | min_gru | single | idle | 1974 | 0.84 | 2.90 / 3.38 | 0 / 180 |
| host | min_gru | single | matmul | 1762 | 0.89 | 2.54 / 2.93 | 0 / 180 |
| host | min_gru | actors | idle | 3039 | 3.87 | 5.24 / 11.80 | 0 / 1440 |
| host | min_gru | actors | matmul | 2730 | 4.03 | 5.23 / 11.99 | 0 / 1440 |
| host | min_gru | batch | idle | 11449 | 0.81 | 3.00 / 5.12 | 0 / 180 |
| host | min_gru | batch | matmul | 10457 | 0.93 | 3.00 / 4.93 | 0 / 180 |
| host | gru | single | idle | 2241 | 0.51 | 2.82 / 3.42 | 0 / 180 |
| host | gru | single | matmul | 2013 | 0.64 | 2.55 / 3.06 | 0 / 180 |
| host | gru | actors | idle | 2368 | 4.15 | 5.87 / 6.49 | 0 / 1440 |
| host | gru | actors | matmul | 2302 | 4.20 | 6.03 / 6.34 | 0 / 1440 |
| host | gru | batch | idle | 10343 | 1.07 | 3.09 / 3.68 | 0 / 180 |
| host | gru | batch | matmul | 9714 | 1.04 | 2.95 / 3.54 | 0 / 180 |
| host | mamba | single | idle | 1080 | 1.03 | 3.03 / 11.41 | 0 / 180 |
| host | mamba | single | matmul | 994 | 1.05 | 3.20 / 12.17 | 0 / 180 |
| host | mamba | actors | idle | 1068 | 6.28 | 68.91 / 81.00 | 280 / 1440 |
| host | mamba | actors | matmul | 1053 | 6.05 | 70.91 / 82.86 | 296 / 1440 |
| host | mamba | batch | idle | 3487 | 3.09 | 16.46 / 32.46 | 9 / 180 |
| host | mamba | batch | matmul | 3329 | 3.12 | 16.15 / 32.15 | 9 / 180 |
| host | mamba2 | single | idle | 1375 | 0.89 | 3.14 / 5.67 | 0 / 180 |
| host | mamba2 | single | matmul | 1188 | 1.15 | 2.88 / 4.18 | 0 / 180 |
| host | mamba2 | actors | idle | 2147 | 8.73 | 10.83 / 25.82 | 48 / 1440 |
| host | mamba2 | actors | matmul | 1955 | 9.36 | 12.22 / 26.48 | 48 / 1440 |
| host | mamba2 | batch | idle | 5623 | 1.89 | 3.83 / 14.84 | 0 / 180 |
| host | mamba2 | batch | matmul | 5249 | 1.93 | 3.61 / 15.39 | 1 / 180 |
| cuda | min_gru | single | idle | 736 | 1.92 | 3.52 / 4.80 | 0 / 180 |
| cuda | min_gru | single | matmul | 156 | 7.21 | 8.60 / 10.21 | 0 / 180 |
| cuda | min_gru | actors | idle | 822 | 12.89 | 16.43 / 24.77 | 68 / 1440 |
| cuda | min_gru | actors | matmul | 508 | 18.63 | 30.58 / 31.71 | 814 / 1440 |
| cuda | min_gru | batch | idle | 7573 | 1.39 | 3.41 / 4.59 | 0 / 180 |
| cuda | min_gru | batch | matmul | 3755 | 2.51 | 4.14 / 6.16 | 0 / 180 |
| cuda | gru | single | idle | 909 | 1.57 | 3.47 / 3.89 | 0 / 180 |
| cuda | gru | single | matmul | 234 | 4.88 | 6.43 / 6.93 | 0 / 180 |
| cuda | gru | actors | idle | 1265 | 7.73 | 9.15 / 9.67 | 0 / 1440 |
| cuda | gru | actors | matmul | 818 | 11.31 | 12.35 / 13.22 | 0 / 1440 |
| cuda | gru | batch | idle | 7812 | 1.33 | 3.62 / 4.08 | 0 / 180 |
| cuda | gru | batch | matmul | 3773 | 2.42 | 4.13 / 4.33 | 0 / 180 |
| cuda | mamba | single | idle | 892 | 1.54 | 3.40 / 6.83 | 0 / 180 |
| cuda | mamba | single | matmul | 231 | 5.39 | 6.55 / 10.73 | 0 / 180 |
| cuda | mamba | actors | idle | 1061 | 8.33 | 25.51 / 36.43 | 102 / 1440 |
| cuda | mamba | actors | matmul | 771 | 11.01 | 34.08 / 41.52 | 192 / 1440 |
| cuda | mamba | batch | idle | 6442 | 1.41 | 3.55 / 9.03 | 0 / 180 |
| cuda | mamba | batch | matmul | 3403 | 2.54 | 4.25 / 9.75 | 0 / 180 |
| cuda | mamba2 | single | idle | 647 | 2.27 | 4.13 / 5.91 | 0 / 180 |
| cuda | mamba2 | single | matmul | 158 | 7.01 | 8.75 / 11.15 | 0 / 180 |
| cuda | mamba2 | actors | idle | 836 | 11.10 | 21.88 / 30.33 | 96 / 1440 |
| cuda | mamba2 | actors | matmul | 547 | 16.21 | 33.67 / 36.33 | 421 / 1440 |
| cuda | mamba2 | batch | idle | 6375 | 1.50 | 3.45 / 5.78 | 0 / 180 |
| cuda | mamba2 | batch | matmul | 3544 | 2.53 | 4.14 / 7.90 | 0 / 180 |
