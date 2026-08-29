# Coach Report — 20260829_120250

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 5 |
| Death sequences | 3 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 5 | 1 | 7 | 5 | 3 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:4177 — passive_run=313f mid_dist=8.3
- **passivity_window** r5.slp:5625 — passive_run=326f mid_dist=9.9
- **passivity_window** r5.slp:5957 — passive_run=349f mid_dist=8.3
- **dropped_punish** r5.slp:1240 — opening@1240 start_pct=42.3 window=120f
- **dropped_punish** r5.slp:1527 — opening@1527 start_pct=49.3 window=120f
- **dropped_punish** r5.slp:1775 — opening@1775 start_pct=55.7 window=120f
- **dropped_punish** r5.slp:2758 — opening@2758 start_pct=111.2 window=120f
- **dropped_punish** r5.slp:3297 — opening@3297 start_pct=120.2 window=120f
- **neutral_loss** r5.slp:463 — hit@553 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r5.slp:2548 — hit@2638 opener=p2_action57 dist_90f_prior=47.1

Appended 10 new gap(s) to `scenarios/gaps.json` (1389 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
