# Coach Report — 20260829_224711

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 5 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 0 |
| Death sequences | 3 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T221001.slp | 0.0 | 5 | 0 | 8 | 0 | 3 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T221001.slp:2578 — passive_run=316f mid_dist=32.7
- **passivity_window** Game_20260829T221001.slp:3708 — passive_run=354f mid_dist=15.9
- **neutral_loss** Game_20260829T221001.slp:313 — hit@403 opener=p2_action68 dist_90f_prior=16.5
- **neutral_loss** Game_20260829T221001.slp:794 — hit@884 opener=p2_action213 dist_90f_prior=27.3
- **neutral_loss** Game_20260829T221001.slp:1200 — hit@1290 opener=p2_action65 dist_90f_prior=53.5
- **neutral_loss** Game_20260829T221001.slp:1965 — hit@2055 opener=p2_action68 dist_90f_prior=48.3
- **neutral_loss** Game_20260829T221001.slp:2317 — hit@2407 opener=p2_action215 dist_90f_prior=25.5
- **neutral_loss** Game_20260829T221001.slp:2690 — hit@2780 opener=p2_action50 dist_90f_prior=16.4
- **neutral_loss** Game_20260829T221001.slp:3598 — hit@3688 opener=p2_action44 dist_90f_prior=32.8
- **neutral_loss** Game_20260829T221001.slp:4030 — hit@4120 opener=p2_action63 dist_90f_prior=18.3

Appended 10 new gap(s) to `scenarios/gaps.json` (1792 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
