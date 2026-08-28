# Coach Report — 20260828_123228

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 0 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 0 | 0 | 8 | 5 | 2 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:323 — passive_run=414f mid_dist=16.0
- **passivity_window** r1.slp:4832 — passive_run=426f mid_dist=8.3
- **passivity_window** r1.slp:5286 — passive_run=537f mid_dist=8.3
- **passivity_window** r1.slp:6366 — passive_run=558f mid_dist=8.3
- **dropped_punish** r1.slp:1176 — opening@1176 start_pct=52.7 window=120f
- **dropped_punish** r1.slp:1471 — opening@1471 start_pct=60.0 window=120f
- **dropped_punish** r1.slp:2662 — opening@2662 start_pct=100.6 window=120f
- **dropped_punish** r1.slp:4340 — opening@4340 start_pct=187.6 window=120f
- **dropped_punish** r1.slp:5991 — opening@5991 start_pct=267.1 window=120f
- **neutral_loss** r1.slp:1228 — hit@1318 opener=p2_action53 dist_90f_prior=59.2

Appended 10 new gap(s) to `scenarios/gaps.json` (731 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
