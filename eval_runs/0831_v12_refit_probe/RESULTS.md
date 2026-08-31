# Teacher-forced coincidence probe

Same 8 expert files (fox-resolved ports), offstage decision
states, all checkpoints see IDENTICAL states — live-play state-distribution
confounds removed. "up" = main_y buckets 12..16. R_state =
state-mediated coincidence (trunk); L_cond = the AR head's wiring lift
(teacher-forced B on/off, main_x forced neutral); predicted offstage
P(up|B)/P(up) ~ R_state x L_cond.

| checkpoint | head | states | mean P(B) % | mean P(up) % | R_state | L_cond | predicted lift |
|---|---|---:|---:|---:|---:|---:|---:|
| v12_ARrefit | autoregressive | 4000 | 41.70 | 12.54 | 1.09 | 2.39 | 2.60 |
| v12_INDrefit | independent | 4000 | 41.24 | 21.01 | 1.13 | — | 1.13 |
| v11_AR | autoregressive | 4000 | 27.16 | 18.14 | 1.05 | 1.14 | 1.20 |

Reference points: 8a live lift AR 2.20-2.65x vs IND 1.05x (frozen trunk);
post-unfreeze ARM replays AR 1.28x vs IND 1.93x (state-confounded, small n).
