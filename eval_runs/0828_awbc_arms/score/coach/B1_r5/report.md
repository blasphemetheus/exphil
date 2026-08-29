# Coach Report — 20260829_120243

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 5 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.49 | 2 | 1 | 7 | 2 | 1 | 5 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:3415 — passive_run=389f mid_dist=8.3
- **passivity_window** r5.slp:4023 — passive_run=421f mid_dist=8.3
- **passivity_window** r5.slp:4450 — passive_run=455f mid_dist=8.3
- **passivity_window** r5.slp:4911 — passive_run=518f mid_dist=8.3
- **passivity_window** r5.slp:6102 — passive_run=340f mid_dist=15.4
- **dropped_punish** r5.slp:1661 — opening@1661 start_pct=107.7 window=120f
- **dropped_punish** r5.slp:1958 — opening@1958 start_pct=126.3 window=120f
- **neutral_loss** r5.slp:1716 — hit@1806 opener=p2_action60 dist_90f_prior=61.3
- **neutral_loss** r5.slp:2021 — hit@2111 opener=p2_action53 dist_90f_prior=59.7
- **neutral_loss** r5.slp:2691 — hit@2781 opener=p2_action44 dist_90f_prior=15.4

Appended 10 new gap(s) to `scenarios/gaps.json` (1309 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
