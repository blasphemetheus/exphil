# Coach Report — 20260828_124553

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 2 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 3 | 0 | 8 | 2 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:3275 — passive_run=323f mid_dist=26.1
- **passivity_window** r2.slp:6423 — passive_run=314f mid_dist=13.0
- **dropped_punish** r2.slp:2065 — opening@2065 start_pct=48.4 window=120f
- **dropped_punish** r2.slp:5159 — opening@5159 start_pct=172.1 window=120f
- **neutral_loss** r2.slp:355 — hit@445 opener=p2_action57 dist_90f_prior=26.6
- **neutral_loss** r2.slp:1306 — hit@1396 opener=p2_action44 dist_90f_prior=12.2
- **neutral_loss** r2.slp:2858 — hit@2948 opener=p2_action257 dist_90f_prior=11.3
- **neutral_loss** r2.slp:3496 — hit@3586 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r2.slp:4803 — hit@4893 opener=p2_action44 dist_90f_prior=68.9
- **neutral_loss** r2.slp:5207 — hit@5297 opener=p2_action57 dist_90f_prior=56.4

Appended 10 new gap(s) to `scenarios/gaps.json` (791 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
