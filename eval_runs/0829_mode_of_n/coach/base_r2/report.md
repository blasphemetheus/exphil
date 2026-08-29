# Coach Report — 20260829_150152

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 2 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.98 | 4 | 3 | 8 | 2 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:6184 — passive_run=361f mid_dist=8.3
- **dropped_punish** r2.slp:2226 — opening@2226 start_pct=69.5 window=120f
- **dropped_punish** r2.slp:4344 — opening@4344 start_pct=161.8 window=120f
- **neutral_loss** r2.slp:1238 — hit@1328 opener=p2_action44 dist_90f_prior=49.0
- **neutral_loss** r2.slp:2758 — hit@2848 opener=p2_action44 dist_90f_prior=14.0
- **neutral_loss** r2.slp:3410 — hit@3500 opener=p2_action53 dist_90f_prior=27.7
- **neutral_loss** r2.slp:3919 — hit@4009 opener=p2_action44 dist_90f_prior=0.1
- **neutral_loss** r2.slp:4856 — hit@4946 opener=p2_action44 dist_90f_prior=15.8
- **neutral_loss** r2.slp:5386 — hit@5476 opener=p2_action356 dist_90f_prior=81.0
- **neutral_loss** r2.slp:5827 — hit@5917 opener=p2_action57 dist_90f_prior=76.5

Appended 10 new gap(s) to `scenarios/gaps.json` (1519 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
