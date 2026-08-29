# Coach Report — 20260829_120251

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 9 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r6.slp | 0.49 | 9 | 2 | 10 | 5 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r6.slp:2064 — passive_run=322f mid_dist=8.3
- **dropped_punish** r6.slp:1909 — opening@1909 start_pct=59.2 window=120f
- **dropped_punish** r6.slp:3630 — opening@3630 start_pct=27.6 window=120f
- **dropped_punish** r6.slp:5364 — opening@5364 start_pct=89.2 window=120f
- **dropped_punish** r6.slp:6372 — opening@6372 start_pct=137.0 window=120f
- **dropped_punish** r6.slp:6741 — opening@6741 start_pct=143.0 window=120f
- **neutral_loss** r6.slp:358 — hit@448 opener=p2_action44 dist_90f_prior=36.3
- **neutral_loss** r6.slp:832 — hit@922 opener=p2_action44 dist_90f_prior=54.2
- **neutral_loss** r6.slp:1519 — hit@1609 opener=p2_action45 dist_90f_prior=70.0
- **neutral_loss** r6.slp:2282 — hit@2372 opener=p2_action44 dist_90f_prior=5.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1399 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
