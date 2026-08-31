# Coach Report — 20260831_141539

Set: 13 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.29 |
| Approaches (total) | 109 |
| Conversions (total) | 14 |
| Neutral losses (opened up) | 103 |
| Dropped punishes | 25 |
| Death sequences | 50 |
| Passivity windows | 18 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260831T133730.slp | 1.33 | 9 | 0 | 12 | 0 | 4 | 1 |
| Game_20260831T133906.slp | 0.0 | 7 | 1 | 5 | 3 | 4 | 3 |
| Game_20260831T134108.slp | 0.0 | 7 | 0 | 3 | 1 | 4 | 1 |
| Game_20260831T134222.slp | 3.78 | 9 | 1 | 7 | 3 | 4 | 1 |
| Game_20260831T134347.slp | 1.66 | 4 | 1 | 8 | 0 | 4 | 0 |
| Game_20260831T134506.slp | 1.33 | 14 | 2 | 7 | 1 | 3 | 0 |
| Game_20260831T134642.slp | 0.54 | 6 | 1 | 10 | 2 | 4 | 3 |
| Game_20260831T134841.slp | 0.87 | 8 | 1 | 6 | 1 | 4 | 1 |
| Game_20260831T134956.slp | 1.93 | 12 | 3 | 9 | 5 | 4 | 3 |
| Game_20260831T135207.slp | 0.45 | 6 | 1 | 11 | 2 | 4 | 3 |
| Game_20260831T135427.slp | 1.19 | 12 | 1 | 10 | 3 | 4 | 0 |
| Game_20260831T135614.slp | 1.99 | 8 | 2 | 11 | 1 | 4 | 2 |
| Game_20260831T135821.slp | 1.66 | 7 | 0 | 4 | 3 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260831T134108.slp:542 — passive_run=351f mid_dist=14.0
- **passivity_window** Game_20260831T134956.slp:589 — passive_run=488f mid_dist=9.5
- **passivity_window** Game_20260831T134841.slp:596 — passive_run=309f mid_dist=1.8
- **passivity_window** Game_20260831T135614.slp:1399 — passive_run=410f mid_dist=21.1
- **passivity_window** Game_20260831T133906.slp:1461 — passive_run=339f mid_dist=17.3
- **passivity_window** Game_20260831T134642.slp:1846 — passive_run=323f mid_dist=21.5
- **passivity_window** Game_20260831T134956.slp:2320 — passive_run=325f mid_dist=0.0
- **passivity_window** Game_20260831T133906.slp:2527 — passive_run=364f mid_dist=2.6
- **passivity_window** Game_20260831T133730.slp:2610 — passive_run=333f mid_dist=6.4
- **passivity_window** Game_20260831T135207.slp:2957 — passive_run=672f mid_dist=13.2

Appended 10 new gap(s) to `scenarios/gaps.json` (1842 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
