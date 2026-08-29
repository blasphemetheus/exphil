# Coach Report — 20260829_150202

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 14 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r6.slp | 0.0 | 1 | 0 | 14 | 0 | 4 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r6.slp:901 — passive_run=332f mid_dist=6.9
- **passivity_window** r6.slp:2012 — passive_run=355f mid_dist=6.7
- **passivity_window** r6.slp:2436 — passive_run=774f mid_dist=35.8
- **neutral_loss** r6.slp:392 — hit@482 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r6.slp:744 — hit@834 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r6.slp:1100 — hit@1190 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r6.slp:1434 — hit@1524 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r6.slp:1833 — hit@1923 opener=p2_action44 dist_90f_prior=16.1
- **neutral_loss** r6.slp:2165 — hit@2255 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r6.slp:2671 — hit@2761 opener=p2_action60 dist_90f_prior=20.0

Appended 10 new gap(s) to `scenarios/gaps.json` (1619 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
