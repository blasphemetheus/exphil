# Coach Report — 20260828_124554

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 7 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 13 |
| Dropped punishes | 1 |
| Death sequences | 3 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 7 | 1 | 13 | 1 | 3 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:3939 — passive_run=303f mid_dist=28.8
- **dropped_punish** r3.slp:5775 — opening@5775 start_pct=18.0 window=120f
- **neutral_loss** r3.slp:798 — hit@888 opener=p2_action44 dist_90f_prior=33.3
- **neutral_loss** r3.slp:1125 — hit@1215 opener=p2_action48 dist_90f_prior=24.5
- **neutral_loss** r3.slp:1469 — hit@1559 opener=p2_action44 dist_90f_prior=40.7
- **neutral_loss** r3.slp:2108 — hit@2198 opener=p2_action60 dist_90f_prior=49.7
- **neutral_loss** r3.slp:2437 — hit@2527 opener=p2_action44 dist_90f_prior=40.2
- **neutral_loss** r3.slp:2882 — hit@2972 opener=p2_action60 dist_90f_prior=88.6
- **neutral_loss** r3.slp:3256 — hit@3346 opener=p2_action60 dist_90f_prior=23.7
- **neutral_loss** r3.slp:4067 — hit@4157 opener=p2_action57 dist_90f_prior=15.7

Appended 10 new gap(s) to `scenarios/gaps.json` (801 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
