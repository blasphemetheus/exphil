# Coach Report — 20260829_120241

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 2 | 1 | 8 | 1 | 4 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:4824 — passive_run=323f mid_dist=26.4
- **passivity_window** r2.slp:5153 — passive_run=416f mid_dist=8.3
- **dropped_punish** r2.slp:4596 — opening@4596 start_pct=134.8 window=120f
- **neutral_loss** r2.slp:365 — hit@455 opener=p2_action44 dist_90f_prior=9.1
- **neutral_loss** r2.slp:1367 — hit@1457 opener=p2_action45 dist_90f_prior=6.7
- **neutral_loss** r2.slp:1841 — hit@1931 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r2.slp:2580 — hit@2670 opener=p2_action60 dist_90f_prior=33.4
- **neutral_loss** r2.slp:3481 — hit@3571 opener=p2_action57 dist_90f_prior=68.6
- **neutral_loss** r2.slp:3828 — hit@3918 opener=p2_action53 dist_90f_prior=68.8
- **neutral_loss** r2.slp:4946 — hit@5036 opener=p2_action44 dist_90f_prior=26.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1279 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
