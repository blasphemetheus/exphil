# Coach Report — 20260826_123831

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.49 | 2 | 0 | 9 | 2 | 1 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:4481 — passive_run=359f mid_dist=23.0
- **passivity_window** r2.slp:5006 — passive_run=425f mid_dist=8.3
- **passivity_window** r2.slp:5550 — passive_run=663f mid_dist=8.3
- **passivity_window** r2.slp:6773 — passive_run=302f mid_dist=11.7
- **dropped_punish** r2.slp:1239 — opening@1239 start_pct=95.9 window=120f
- **dropped_punish** r2.slp:2848 — opening@2848 start_pct=163.2 window=120f
- **neutral_loss** r2.slp:866 — hit@956 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r2.slp:1492 — hit@1582 opener=p2_action57 dist_90f_prior=36.7
- **neutral_loss** r2.slp:1823 — hit@1913 opener=p2_action60 dist_90f_prior=22.6
- **neutral_loss** r2.slp:2414 — hit@2504 opener=p2_action44 dist_90f_prior=27.5

Appended 10 new gap(s) to `scenarios/gaps.json` (276 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
