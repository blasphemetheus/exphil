# Drill bank score — eval_runs/0902_drill1_bank2 (uthrow_low_mid)

150 episodes scored (window 240 f from the recorded handoff; detector =
drill_table_mine's hitstun/thrown/captured rising edges). Anchor mismatches
(port-1 action not a throw at handoff): 0. Live-counter
disagreements: 92/150.

| set | n | mean hits | >=3 hits % | mean dmg | stocks |
|---|---:|---:|---:|---:|---:|
| bot 0-19% | 96 | 2.7 | 30 | 16.3 | 0 |
| expert 0-19% | 39 | 3.9 | 87 | 27.0 | 0 |
| bot 20-39% | 54 | 1.8 | 11 | 13.5 | 1 |
| expert 20-39% | 24 | 3.7 | 83 | 20.5 | 0 |


0-19% hits histogram: 1:24  2:43  3:14  4:4  6:1  7:3  8:2  9:4  11:1


20-39% hits histogram: 1:21  2:27  3:5  8:1


## Bank notes (2026-09-02, first laddered bank)

- First run on the manifest + band-laddering driver: 150 episodes across
  14 games (~11 eps/game vs 4 pre-laddering), 0-19: 96 / 20-39: 54.
- 0-19 reproduces the 388-ep baseline (2.7/30/16.3 vs 2.9/30/16.4) —
  the drill measurement is stable across driver versions.
- 20-39 is WORSE than 0-19 (1.8 hits / 11% deep vs expert's 3.7/83) —
  the uthrow follow-up at rising percent (higher knockback, later uair
  timing) is even less present than the low-percent confirm.
- Combined Drill 1 corpus for the retrain arm: 538 episodes
  (0902_drill1_bank 388 + this bank 150), slice-by-window per drill.json.
