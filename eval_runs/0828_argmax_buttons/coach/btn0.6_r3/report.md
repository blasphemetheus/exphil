# Coach Report — 20260828_162817

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 3 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 5 | 1 | 8 | 3 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:2714 — passive_run=362f mid_dist=8.3
- **dropped_punish** r3.slp:403 — opening@403 start_pct=17.0 window=120f
- **dropped_punish** r3.slp:652 — opening@652 start_pct=17.0 window=120f
- **dropped_punish** r3.slp:5095 — opening@5095 start_pct=0.0 window=120f
- **neutral_loss** r3.slp:484 — hit@574 opener=p2_action60 dist_90f_prior=27.8
- **neutral_loss** r3.slp:1100 — hit@1190 opener=p2_action44 dist_90f_prior=21.0
- **neutral_loss** r3.slp:1619 — hit@1709 opener=p2_action57 dist_90f_prior=6.8
- **neutral_loss** r3.slp:2363 — hit@2453 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r3.slp:4076 — hit@4166 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r3.slp:4451 — hit@4541 opener=p2_action44 dist_90f_prior=6.9

Appended 10 new gap(s) to `scenarios/gaps.json` (1021 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
