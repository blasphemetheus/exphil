# Coach Report — 20260829_120259

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 7 |
| Death sequences | 4 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r7.slp | 0.49 | 3 | 2 | 12 | 7 | 4 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r7.slp:2175 — passive_run=319f mid_dist=8.3
- **passivity_window** r7.slp:2659 — passive_run=360f mid_dist=6.7
- **passivity_window** r7.slp:4640 — passive_run=312f mid_dist=2.6
- **dropped_punish** r7.slp:393 — opening@393 start_pct=55.3 window=120f
- **dropped_punish** r7.slp:752 — opening@752 start_pct=77.6 window=120f
- **dropped_punish** r7.slp:3640 — opening@3640 start_pct=207.6 window=120f
- **dropped_punish** r7.slp:4469 — opening@4469 start_pct=22.3 window=120f
- **dropped_punish** r7.slp:5479 — opening@5479 start_pct=54.1 window=120f
- **dropped_punish** r7.slp:6228 — opening@6228 start_pct=94.9 window=120f
- **dropped_punish** r7.slp:6612 — opening@6612 start_pct=99.9 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (1489 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
