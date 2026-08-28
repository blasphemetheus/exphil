# Coach Report — 20260828_121905

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 4 |
| Death sequences | 2 |
| Passivity windows | 6 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 2 | 0 | 7 | 4 | 2 | 6 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:1624 — passive_run=400f mid_dist=8.3
- **passivity_window** r3.slp:3050 — passive_run=366f mid_dist=8.3
- **passivity_window** r3.slp:3422 — passive_run=477f mid_dist=8.3
- **passivity_window** r3.slp:4366 — passive_run=457f mid_dist=8.3
- **passivity_window** r3.slp:4829 — passive_run=331f mid_dist=8.3
- **passivity_window** r3.slp:5549 — passive_run=537f mid_dist=8.3
- **dropped_punish** r3.slp:642 — opening@642 start_pct=54.6 window=120f
- **dropped_punish** r3.slp:1189 — opening@1189 start_pct=95.2 window=120f
- **dropped_punish** r3.slp:6370 — opening@6370 start_pct=311.9 window=120f
- **dropped_punish** r3.slp:6713 — opening@6713 start_pct=316.9 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (701 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
