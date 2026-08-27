# Coach Report — 20260827_180449

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 2 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 3 | 2 | 7 | 2 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:3945 — passive_run=631f mid_dist=8.3
- **dropped_punish** r2.slp:2423 — opening@2423 start_pct=116.8 window=120f
- **dropped_punish** r2.slp:3385 — opening@3385 start_pct=160.2 window=120f
- **neutral_loss** r2.slp:572 — hit@662 opener=p2_action257 dist_90f_prior=8.3
- **neutral_loss** r2.slp:1269 — hit@1359 opener=p2_action60 dist_90f_prior=11.4
- **neutral_loss** r2.slp:1589 — hit@1679 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r2.slp:3492 — hit@3582 opener=p2_action80 dist_90f_prior=54.8
- **neutral_loss** r2.slp:4494 — hit@4584 opener=p2_action44 dist_90f_prior=1.2
- **neutral_loss** r2.slp:6266 — hit@6356 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r2.slp:6701 — hit@6791 opener=p2_action53 dist_90f_prior=65.3

Appended 10 new gap(s) to `scenarios/gaps.json` (545 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
