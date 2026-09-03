# Drill bank score — eval_runs/0902_awbc_drill/drill_remeasure (uthrow_low_mid)

150 episodes scored (window 240 f from the recorded handoff; detector =
drill_table_mine's hitstun/thrown/captured rising edges). Anchor mismatches
(port-1 action not a throw at handoff): 0. Live-counter
disagreements: 113/150.

| set | n | mean hits | >=3 hits % | mean dmg | stocks |
|---|---:|---:|---:|---:|---:|
| bot 0-19% | 88 | 3.7 | 41 | 17.5 | 0 |
| expert 0-19% | 39 | 3.9 | 87 | 27.0 | 0 |
| bot 20-39% | 62 | 2.5 | 23 | 12.3 | 0 |
| expert 20-39% | 24 | 3.7 | 83 | 20.5 | 0 |


0-19% hits histogram: 1:23  2:29  3:9  4:1  6:1  7:2  8:17  9:5  10:1


20-39% hits histogram: 1:28  2:20  3:5  7:4  8:4  10:1


## Verdict vs the pre-registered gate (baseline = 538-ep bank)

| | mean hits | >=3 % | mean dmg | 1-2 hit share | 8+ hit share |
|---|---:|---:|---:|---:|---:|
| baseline (0-19) | 2.9 | 30 | 16.4 | 70% | 12% |
| AWBCdrill (0-19) | 3.7 | 41 | 17.5 | 59% | 26% |
| expert (0-19) | 3.9 | 87 | 27.0 | — | — |

PARTIAL PASS, with a confound: the distribution DID shift (gate metric:
mean hits 2.9 -> 3.7, deep3 30 -> 41, 1-2-hit share 70 -> 59) — but the
biggest single change is the 8+-hit LASER-STRING mode doubling (12 ->
26%), and mean damage barely moved (16.4 -> 17.5 vs expert 27.0). If the
canonical uthrow->uair chains had arrived, damage would move toward 27;
lasers give hit-count at low damage. AWBC's damage-RTG upweighted the
continuation class the policy could already reach (laser strings) more
than the one we wanted (aerial chains). 20-39 band: 1.8 -> 2.5 hits,
11 -> 23% deep, dmg 13.5 -> 12.3 (same story).

Next per pre-registration: Bradley live-look transfer check (does
uthrow -> uair -> uair appear in real play?). Knob candidates if the
live look agrees with the laser confound: awbc reward shaping toward
per-hit damage, or drill-cell filtering of laser continuations from the
mix (both one-knob changes, neither pre-registered yet).
