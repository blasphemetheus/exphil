# Coach Report — 20260827_181710

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 7 |
| Conversions (total) | 5 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 8 |
| Death sequences | 2 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 7 | 5 | 8 | 8 | 2 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:3804 — passive_run=450f mid_dist=20.3
- **passivity_window** r1.slp:4538 — passive_run=309f mid_dist=7.0
- **passivity_window** r1.slp:6597 — passive_run=338f mid_dist=13.7
- **dropped_punish** r1.slp:330 — opening@330 start_pct=15.5 window=120f
- **dropped_punish** r1.slp:1517 — opening@1517 start_pct=51.6 window=120f
- **dropped_punish** r1.slp:1873 — opening@1873 start_pct=59.8 window=120f
- **dropped_punish** r1.slp:2354 — opening@2354 start_pct=84.4 window=120f
- **dropped_punish** r1.slp:5015 — opening@5015 start_pct=184.6 window=120f
- **dropped_punish** r1.slp:5825 — opening@5825 start_pct=189.6 window=120f
- **dropped_punish** r1.slp:6156 — opening@6156 start_pct=204.6 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (584 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
