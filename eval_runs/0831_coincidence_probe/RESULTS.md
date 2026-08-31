# Teacher-forced coincidence probe

Same 8 expert files (fox-resolved ports), offstage decision
states, all checkpoints see IDENTICAL states — live-play state-distribution
confounds removed. "up" = main_y buckets 12..16. R_state =
state-mediated coincidence (trunk); L_cond = the AR head's wiring lift
(teacher-forced B on/off, main_x forced neutral); predicted offstage
P(up|B)/P(up) ~ R_state x L_cond.

| checkpoint | head | states | mean P(B) % | mean P(up) % | R_state | L_cond | predicted lift |
|---|---|---:|---:|---:|---:|---:|---:|
| v11_AR | autoregressive | 4000 | 27.16 | 18.14 | 1.05 | 1.14 | 1.20 |
| v11_IND | independent | 4000 | 22.88 | 13.84 | 1.10 | — | 1.10 |
| 8a_ARhead | autoregressive | 4000 | 41.46 | 11.94 | 1.07 | 2.67 | 2.87 |
| 8a_INDhead | independent | 4000 | 41.56 | 20.32 | 1.14 | — | 1.14 |

Reference points: 8a live lift AR 2.20-2.65x vs IND 1.05x (frozen trunk);
post-unfreeze ARM replays AR 1.28x vs IND 1.93x (state-confounded, small n).

## Verdict (2026-08-31 14:40)

1. **Instrument validated**: 8a_ARhead predicted 2.87 vs measured live
   2.20–2.65x; v11_AR predicted 1.20 vs measured 1.28x. The live
   inversion was real, not small-n.
2. **The unfreeze ATROPHIED the AR conditioning wire**: L_cond 2.67 →
   1.14 with R_state flat everywhere — the trunk did NOT absorb the
   dependency ("absorption" hypothesis rejected); joint training simply
   stopped expressing it. Frozen-trunk fits FORCE the tf-wire to carry
   the within-frame dependency; an unfrozen trunk offers cheaper global
   loss reductions and the wire fades (shortcut dynamics).
3. **Consequence**: v1.1-AR is effectively near-independent today (its
   0.49-nat val edge must live in other conditional mass — unmeasured
   pairs / c-stick — or plain fit). Cheap restoration: head-only refit
   (the 8a recipe, full loss recipe per L12) on v1.1-AR's improved
   trunk — queued as the head line's next arm. Alternatives if that
   regresses again on any future unfreeze: lower trunk LR for the head
   phase, or freeze-last-epoch schedules.
