# Coach Report — 20260828_162820

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 3 |
| Death sequences | 3 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r6.slp | 0.0 | 1 | 1 | 7 | 3 | 3 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r6.slp:6455 — passive_run=398f mid_dist=6.7
- **dropped_punish** r6.slp:432 — opening@432 start_pct=19.8 window=120f
- **dropped_punish** r6.slp:1316 — opening@1316 start_pct=58.4 window=120f
- **dropped_punish** r6.slp:1691 — opening@1691 start_pct=67.6 window=120f
- **neutral_loss** r6.slp:1934 — hit@2024 opener=p2_action44 dist_90f_prior=6.1
- **neutral_loss** r6.slp:3251 — hit@3341 opener=p2_action60 dist_90f_prior=61.7
- **neutral_loss** r6.slp:3703 — hit@3793 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r6.slp:4231 — hit@4321 opener=p2_action60 dist_90f_prior=53.3
- **neutral_loss** r6.slp:5056 — hit@5146 opener=p2_action60 dist_90f_prior=33.6
- **neutral_loss** r6.slp:5826 — hit@5916 opener=p2_action44 dist_90f_prior=29.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1051 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
