# Stochastic recurrent controller benchmarks — October 26, 2026

All entries below are **median / p95 milliseconds for the whole request**, not per row.

## Steady recurrent inference

| Policy | CPU, 1 policy | CPU, 8 policies | CUDA, 1 policy | CUDA, 8 policies |
|---|---:|---:|---:|---:|
| MinGRU | 0.558 / 0.788 | 0.742 / 0.903 | 3.717 / 4.422 | 3.573 / 4.153 |
| GRU | 0.517 / 0.683 | 1.005 / 1.329 | 3.838 / 4.308 | 3.727 / 4.201 |
| Mamba | 0.768 / 1.147 | 1.396 / 1.755 | 3.557 / 4.055 | 3.742 / 4.040 |
| Mamba-2 | 0.690 / 1.089 | 1.359 / 1.785 | 3.658 / 4.111 | 2.991 / 3.500 |

## Deterministic versus stochastic, CUDA

| Policy | Rows | Deterministic | Stochastic |
|---|---:|---:|---:|
| MinGRU | 1 | 1.450 / 1.825 | 3.717 / 4.422 |
| MinGRU | 8 | 0.909 / 1.101 | 3.573 / 4.153 |
| GRU | 1 | 0.920 / 1.149 | 3.838 / 4.308 |
| GRU | 8 | 0.997 / 1.208 | 3.727 / 4.201 |
| Mamba | 1 | 1.226 / 1.615 | 3.557 / 4.055 |
| Mamba | 8 | 1.057 / 1.358 | 3.742 / 4.040 |
| Mamba-2 | 1 | 1.447 / 2.094 | 3.658 / 4.111 |
| Mamba-2 | 8 | 0.944 / 1.226 | 2.991 / 3.500 |

## Eight-policy stochastic replay and full-window costs

| Policy | Backend | Reset 1 row | Reset all | Resync 1 row | Resync all | Windowed |
|---|---|---:|---:|---:|---:|---:|
| MinGRU | host | 1.732 / 1.987 | 2.603 / 3.433 | 1.795 / 2.089 | 2.465 / 3.079 | 3.156 / 6.320 |
| MinGRU | cuda | 5.283 / 5.853 | 5.224 / 5.623 | 5.309 / 5.771 | 5.281 / 5.792 | 4.324 / 4.845 |
| GRU | host | 0.925 / 1.157 | 0.838 / 1.110 | — | — | — |
| GRU | cuda | 3.691 / 4.102 | 3.495 / 3.825 | — | — | — |
| Mamba | host | 11.412 / 13.302 | 29.634 / 37.920 | 11.718 / 12.877 | 30.996 / 34.908 | 153.856 / 165.661 |
| Mamba | cuda | 7.136 / 7.659 | 9.031 / 9.415 | 7.385 / 7.764 | 9.258 / 9.740 | 4.966 / 5.426 |
| Mamba-2 | host | 4.339 / 4.795 | 9.989 / 11.958 | 4.605 / 6.930 | 10.128 / 12.985 | 41.620 / 47.313 |
| Mamba-2 | cuda | 5.401 / 5.859 | 5.879 / 6.388 | 5.437 / 5.926 | 6.047 / 6.676 | 3.609 / 4.053 |

GRU is BPTT-trained: its valid deployment path retains carry. Periodic resync and
windowed recomputation are deliberately absent.
