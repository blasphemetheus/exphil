# Coach Report — 20260828_162825

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 3 | 0 | 8 | 0 | 4 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:876 — passive_run=440f mid_dist=6.8
- **passivity_window** r5.slp:1361 — passive_run=823f mid_dist=20.0
- **passivity_window** r5.slp:2566 — passive_run=825f mid_dist=8.7
- **neutral_loss** r5.slp:307 — hit@397 opener=p2_action44 dist_90f_prior=3.1
- **neutral_loss** r5.slp:715 — hit@805 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r5.slp:1099 — hit@1189 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r5.slp:1497 — hit@1587 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r5.slp:1991 — hit@2081 opener=p2_action44 dist_90f_prior=7.5
- **neutral_loss** r5.slp:2378 — hit@2468 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r5.slp:2720 — hit@2810 opener=p2_action44 dist_90f_prior=8.0

Appended 10 new gap(s) to `scenarios/gaps.json` (1081 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
