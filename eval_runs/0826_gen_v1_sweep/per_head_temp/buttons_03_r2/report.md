# Coach Report — 20260827_181711

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 2 | 0 | 7 | 5 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:2859 — passive_run=422f mid_dist=8.3
- **dropped_punish** r2.slp:386 — opening@386 start_pct=42.1 window=120f
- **dropped_punish** r2.slp:2623 — opening@2623 start_pct=112.5 window=120f
- **dropped_punish** r2.slp:3600 — opening@3600 start_pct=144.3 window=120f
- **dropped_punish** r2.slp:5502 — opening@5502 start_pct=4.6 window=120f
- **dropped_punish** r2.slp:6499 — opening@6499 start_pct=28.9 window=120f
- **neutral_loss** r2.slp:1406 — hit@1496 opener=p2_action44 dist_90f_prior=38.8
- **neutral_loss** r2.slp:1833 — hit@1923 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r2.slp:2680 — hit@2770 opener=p2_action44 dist_90f_prior=65.2
- **neutral_loss** r2.slp:3379 — hit@3469 opener=p2_action44 dist_90f_prior=24.3

Appended 10 new gap(s) to `scenarios/gaps.json` (594 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
