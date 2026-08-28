# Coach Report — 20260828_124556

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 6 |
| Death sequences | 1 |
| Passivity windows | 5 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 4 | 1 | 12 | 6 | 1 | 5 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:1386 — passive_run=400f mid_dist=13.8
- **passivity_window** r5.slp:1880 — passive_run=327f mid_dist=6.8
- **passivity_window** r5.slp:3340 — passive_run=478f mid_dist=8.3
- **passivity_window** r5.slp:4158 — passive_run=301f mid_dist=13.1
- **passivity_window** r5.slp:5836 — passive_run=352f mid_dist=8.3
- **dropped_punish** r5.slp:668 — opening@668 start_pct=52.7 window=120f
- **dropped_punish** r5.slp:1354 — opening@1354 start_pct=79.3 window=120f
- **dropped_punish** r5.slp:2528 — opening@2528 start_pct=179.8 window=120f
- **dropped_punish** r5.slp:5343 — opening@5343 start_pct=76.8 window=120f
- **dropped_punish** r5.slp:6396 — opening@6396 start_pct=151.1 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (821 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
