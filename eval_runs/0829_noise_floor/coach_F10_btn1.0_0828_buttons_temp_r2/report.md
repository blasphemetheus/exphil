# Coach Report — 20260829_231258

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 2 | 1 | 6 | 2 | 1 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:1293 — passive_run=307f mid_dist=8.3
- **passivity_window** r2.slp:2053 — passive_run=362f mid_dist=8.3
- **passivity_window** r2.slp:5285 — passive_run=428f mid_dist=8.3
- **passivity_window** r2.slp:5719 — passive_run=462f mid_dist=8.3
- **dropped_punish** r2.slp:2756 — opening@2756 start_pct=141.5 window=120f
- **dropped_punish** r2.slp:7352 — opening@7352 start_pct=323.6 window=120f
- **neutral_loss** r2.slp:2294 — hit@2384 opener=p2_action44 dist_90f_prior=0.0
- **neutral_loss** r2.slp:3195 — hit@3285 opener=p2_action44 dist_90f_prior=46.1
- **neutral_loss** r2.slp:3584 — hit@3674 opener=p2_action60 dist_90f_prior=61.3
- **neutral_loss** r2.slp:3880 — hit@3970 opener=p2_action44 dist_90f_prior=6.8

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
