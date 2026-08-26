# Coach Report — 20260826_122126

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 0 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 1 |
| Death sequences | 1 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 0 | 0 | 9 | 1 | 1 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:1403 — passive_run=310f mid_dist=7.0
- **passivity_window** r1.slp:2728 — passive_run=339f mid_dist=26.6
- **passivity_window** r1.slp:3763 — passive_run=506f mid_dist=8.3
- **dropped_punish** r1.slp:5737 — opening@5737 start_pct=37.2 window=120f
- **neutral_loss** r1.slp:1594 — hit@1684 opener=p2_action60 dist_90f_prior=7.0
- **neutral_loss** r1.slp:1948 — hit@2038 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r1.slp:2452 — hit@2542 opener=p2_action44 dist_90f_prior=6.0
- **neutral_loss** r1.slp:2862 — hit@2952 opener=p2_action60 dist_90f_prior=0.7
- **neutral_loss** r1.slp:3658 — hit@3748 opener=p2_action44 dist_90f_prior=49.3
- **neutral_loss** r1.slp:4849 — hit@4939 opener=p2_action44 dist_90f_prior=6.6

Appended 10 new gap(s) to `scenarios/gaps.json` (216 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
