# Teacher-forced coincidence probe

Same 8 expert files (fox-resolved ports), offstage decision
states, all checkpoints see IDENTICAL states — live-play state-distribution
confounds removed. "up" = main_y buckets 12..16. R_state =
state-mediated coincidence (trunk); L_cond = the AR head's wiring lift
(teacher-forced B on/off, main_x forced neutral); predicted offstage
P(up|B)/P(up) ~ R_state x L_cond.

| checkpoint | head | states | mean P(B) % | mean P(up) % | R_state | L_cond | predicted lift |
|---|---|---:|---:|---:|---:|---:|---:|
| v13_ARrefit | autoregressive | 4000 | 38.67 | 21.41 | 1.31 | 1.49 | 1.96 |
| v13_INDrefit | independent | 4000 | 37.90 | 24.91 | 1.40 | — | 1.40 |
| v13_AR | autoregressive | 4000 | 34.01 | 27.71 | 1.19 | 1.09 | 1.30 |

Reference points: 8a live lift AR 2.20-2.65x vs IND 1.05x (frozen trunk);
post-unfreeze ARM replays AR 1.28x vs IND 1.93x (state-confounded, small n).
