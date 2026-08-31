# Fox-only expert-baseline re-reads (post-E1)

**2026-08-30 22:10.** E1 showed the "expert = port 1" baseline was ~43%
opponent characters (`../0830_corpus_mix/RESULTS.md`). These re-reads use
`--expert-char 2` (per-file fox port resolution, dittos skipped) on the
same instruments; bot sets unchanged (AR/IND bracket-2 arms, ep10 CPU).

## Verdict: every conclusion STANDS; the recovery gaps get slightly BIGGER

| expert metric | mixed (old) | fox-only (new) |
|---|---:|---:|
| A2 first route: up-B | 16.3% | **20.6%** |
| A2 first route: double_jump | 24.8% | **28.9%** |
| A2 first route: airdodge | 4.8% | **3.7%** |
| A2 first route: none | 14.6% | **9.9%** |
| A2 recovery died % | 12.7 | 14.3 |
| C4 unforced deaths % | 26.9 | **19.0** |
| C4 edgeguarded % | 50.2 (new col) | – |
| B2 mean TV (AR / IND / ep10) | 0.62 / 0.61 / 0.53 | 0.63 / 0.62 / 0.55 |

- The fox-only expert up-Bs and double-jumps MORE and airdodges LESS than
  the mixed baseline said → the bots' route deficit (and the AR head's
  restoration of up-B) is understated by the old numbers, not an artifact
  of character mixing.
- C4: the fox-only expert dies unforced only 19% of the time (mixed said
  27%) — the bots' 61–65% unforced share is ~3.3× expert, worse than the
  old read. Expert deaths are mostly EDGEGUARDED (50%) at 109%; the bots
  die unhit at ~27%.
- B2 TV shifts by +0.01–0.02 uniformly; ordering unchanged. The mixed
  baseline was not distorting decode rankings.

Files: `situation_hist.md`, `edge_scorecard.md`, `death_classifier.md`.
The `--expert-char` flag is now on situation_hist / edge_scorecard /
death_classifier (punish_quality / reaction_latency had it from birth).
