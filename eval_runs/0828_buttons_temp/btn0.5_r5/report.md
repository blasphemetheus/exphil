# Coach Report — 20260828_125920

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.49 | 3 | 0 | 7 | 5 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:4698 — passive_run=612f mid_dist=8.3
- **passivity_window** r5.slp:5933 — passive_run=460f mid_dist=8.3
- **dropped_punish** r5.slp:332 — opening@332 start_pct=30.5 window=120f
- **dropped_punish** r5.slp:833 — opening@833 start_pct=55.3 window=120f
- **dropped_punish** r5.slp:2668 — opening@2668 start_pct=125.7 window=120f
- **dropped_punish** r5.slp:5280 — opening@5280 start_pct=208.2 window=120f
- **dropped_punish** r5.slp:7265 — opening@7265 start_pct=271.3 window=120f
- **neutral_loss** r5.slp:534 — hit@624 opener=p2_action44 dist_90f_prior=8.4
- **neutral_loss** r5.slp:1526 — hit@1616 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r5.slp:2728 — hit@2818 opener=p2_action60 dist_90f_prior=57.4

Appended 10 new gap(s) to `scenarios/gaps.json` (871 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
