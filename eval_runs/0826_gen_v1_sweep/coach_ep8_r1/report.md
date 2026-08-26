# Coach Report — 20260826_132657

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.47 |
| Approaches (total) | 6 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 1 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 1.47 | 6 | 2 | 11 | 1 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:6113 — passive_run=412f mid_dist=26.7
- **passivity_window** r1.slp:7007 — passive_run=320f mid_dist=6.7
- **dropped_punish** r1.slp:2320 — opening@2320 start_pct=115.0 window=120f
- **neutral_loss** r1.slp:406 — hit@496 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r1.slp:937 — hit@1027 opener=p2_action60 dist_90f_prior=41.5
- **neutral_loss** r1.slp:1350 — hit@1440 opener=p2_action57 dist_90f_prior=62.3
- **neutral_loss** r1.slp:1623 — hit@1713 opener=p2_action60 dist_90f_prior=64.6
- **neutral_loss** r1.slp:1988 — hit@2078 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r1.slp:2324 — hit@2414 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r1.slp:3140 — hit@3230 opener=p2_action44 dist_90f_prior=5.2

Appended 10 new gap(s) to `scenarios/gaps.json` (445 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
