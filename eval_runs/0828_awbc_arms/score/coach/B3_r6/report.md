# Coach Report — 20260829_120258

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 4 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 3 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r6.slp | 0.49 | 4 | 2 | 7 | 3 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r6.slp:4922 — passive_run=369f mid_dist=5.3
- **passivity_window** r6.slp:7027 — passive_run=322f mid_dist=8.3
- **dropped_punish** r6.slp:464 — opening@464 start_pct=28.2 window=120f
- **dropped_punish** r6.slp:2263 — opening@2263 start_pct=106.6 window=120f
- **dropped_punish** r6.slp:3680 — opening@3680 start_pct=5.0 window=120f
- **neutral_loss** r6.slp:1314 — hit@1404 opener=p2_action57 dist_90f_prior=71.9
- **neutral_loss** r6.slp:3493 — hit@3583 opener=p2_action44 dist_90f_prior=23.7
- **neutral_loss** r6.slp:3754 — hit@3844 opener=p2_action44 dist_90f_prior=30.8
- **neutral_loss** r6.slp:4394 — hit@4484 opener=p2_action60 dist_90f_prior=12.5
- **neutral_loss** r6.slp:5181 — hit@5271 opener=p2_action53 dist_90f_prior=20.2

Appended 10 new gap(s) to `scenarios/gaps.json` (1479 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
