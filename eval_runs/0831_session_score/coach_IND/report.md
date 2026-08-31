# Coach Report — 20260831_141541

Set: 7 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.41 |
| Approaches (total) | 54 |
| Conversions (total) | 7 |
| Neutral losses (opened up) | 27 |
| Dropped punishes | 11 |
| Death sequences | 22 |
| Passivity windows | 5 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260831T140334.slp | 0.78 | 13 | 0 | 3 | 1 | 4 | 0 |
| Game_20260831T140457.slp | 0.0 | 7 | 1 | 1 | 1 | 3 | 0 |
| Game_20260831T140600.slp | 2.78 | 5 | 3 | 3 | 0 | 4 | 0 |
| Game_20260831T140711.slp | 1.63 | 11 | 2 | 6 | 2 | 4 | 3 |
| Game_20260831T140908.slp | 4.24 | 8 | 1 | 3 | 4 | 3 | 0 |
| Game_20260831T141011.slp | 0.45 | 10 | 0 | 11 | 3 | 4 | 2 |
| Game_20260831T141230.slp | 0.0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260831T140711.slp:3486 — passive_run=312f mid_dist=3.0
- **passivity_window** Game_20260831T141011.slp:3938 — passive_run=568f mid_dist=23.1
- **passivity_window** Game_20260831T140711.slp:5233 — passive_run=498f mid_dist=10.7
- **passivity_window** Game_20260831T141011.slp:5365 — passive_run=341f mid_dist=9.8
- **passivity_window** Game_20260831T140711.slp:6172 — passive_run=417f mid_dist=5.2
- **dropped_punish** Game_20260831T140334.slp:301 — opening@301 start_pct=27.0 window=120f
- **dropped_punish** Game_20260831T140908.slp:395 — opening@395 start_pct=2.0 window=120f
- **dropped_punish** Game_20260831T141011.slp:697 — opening@697 start_pct=6.7 window=120f
- **dropped_punish** Game_20260831T140908.slp:947 — opening@947 start_pct=5.8 window=120f
- **dropped_punish** Game_20260831T140711.slp:1176 — opening@1176 start_pct=43.5 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (1852 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
