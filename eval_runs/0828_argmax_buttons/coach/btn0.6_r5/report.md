# Coach Report — 20260828_162819

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 4 | 0 | 6 | 3 | 2 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:2651 — passive_run=343f mid_dist=8.3
- **passivity_window** r5.slp:3255 — passive_run=370f mid_dist=22.1
- **passivity_window** r5.slp:4101 — passive_run=321f mid_dist=8.3
- **passivity_window** r5.slp:4678 — passive_run=376f mid_dist=8.3
- **dropped_punish** r5.slp:1497 — opening@1497 start_pct=62.9 window=120f
- **dropped_punish** r5.slp:1997 — opening@1997 start_pct=77.9 window=120f
- **dropped_punish** r5.slp:5138 — opening@5138 start_pct=194.7 window=120f
- **neutral_loss** r5.slp:468 — hit@558 opener=p2_action60 dist_90f_prior=21.7
- **neutral_loss** r5.slp:874 — hit@964 opener=p2_action44 dist_90f_prior=45.1
- **neutral_loss** r5.slp:2536 — hit@2626 opener=p2_action44 dist_90f_prior=15.2

Appended 10 new gap(s) to `scenarios/gaps.json` (1041 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
