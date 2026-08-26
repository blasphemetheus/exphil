# Coach Report — 20260826_133515

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 0 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 5 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 0 | 0 | 5 | 5 | 1 | 5 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:1318 — passive_run=301f mid_dist=7.1
- **passivity_window** r2.slp:1806 — passive_run=309f mid_dist=8.3
- **passivity_window** r2.slp:3227 — passive_run=544f mid_dist=10.1
- **passivity_window** r2.slp:3777 — passive_run=413f mid_dist=8.3
- **passivity_window** r2.slp:4196 — passive_run=441f mid_dist=8.3
- **dropped_punish** r2.slp:469 — opening@469 start_pct=0.0 window=120f
- **dropped_punish** r2.slp:1655 — opening@1655 start_pct=76.5 window=120f
- **dropped_punish** r2.slp:2854 — opening@2854 start_pct=126.7 window=120f
- **dropped_punish** r2.slp:5161 — opening@5161 start_pct=265.6 window=120f
- **dropped_punish** r2.slp:5691 — opening@5691 start_pct=0.0 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (485 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
