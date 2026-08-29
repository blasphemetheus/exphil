# Coach Report — 20260829_120246

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 7 |
| Death sequences | 1 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.0 | 4 | 1 | 8 | 7 | 1 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r8.slp:4244 — passive_run=367f mid_dist=8.3
- **passivity_window** r8.slp:4648 — passive_run=472f mid_dist=8.3
- **passivity_window** r8.slp:5321 — passive_run=394f mid_dist=8.3
- **dropped_punish** r8.slp:392 — opening@392 start_pct=5.0 window=120f
- **dropped_punish** r8.slp:995 — opening@995 start_pct=65.5 window=120f
- **dropped_punish** r8.slp:1461 — opening@1461 start_pct=75.0 window=120f
- **dropped_punish** r8.slp:2100 — opening@2100 start_pct=106.2 window=120f
- **dropped_punish** r8.slp:2573 — opening@2573 start_pct=115.0 window=120f
- **dropped_punish** r8.slp:3416 — opening@3416 start_pct=160.0 window=120f
- **dropped_punish** r8.slp:5161 — opening@5161 start_pct=213.9 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (1339 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
