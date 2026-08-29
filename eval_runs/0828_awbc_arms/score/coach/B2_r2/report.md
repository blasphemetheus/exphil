# Coach Report — 20260829_120248

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.47 |
| Approaches (total) | 7 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 3 |
| Death sequences | 3 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 1.47 | 7 | 2 | 9 | 3 | 3 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:1142 — passive_run=309f mid_dist=16.5
- **passivity_window** r2.slp:2050 — passive_run=394f mid_dist=15.9
- **passivity_window** r2.slp:6834 — passive_run=355f mid_dist=8.3
- **dropped_punish** r2.slp:3927 — opening@3927 start_pct=178.5 window=120f
- **dropped_punish** r2.slp:6554 — opening@6554 start_pct=115.0 window=120f
- **dropped_punish** r2.slp:7302 — opening@7302 start_pct=154.8 window=120f
- **neutral_loss** r2.slp:477 — hit@567 opener=p2_action44 dist_90f_prior=55.5
- **neutral_loss** r2.slp:882 — hit@972 opener=p2_action44 dist_90f_prior=25.0
- **neutral_loss** r2.slp:1316 — hit@1406 opener=p2_action45 dist_90f_prior=26.8
- **neutral_loss** r2.slp:2346 — hit@2436 opener=p2_action45 dist_90f_prior=19.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1359 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
