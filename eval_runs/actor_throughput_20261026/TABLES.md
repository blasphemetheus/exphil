# Continuous actor throughput screen (partial)

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
