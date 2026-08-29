# Coach Report — 20260829_120247

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 6 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.98 | 6 | 3 | 12 | 5 | 1 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:1420 — passive_run=319f mid_dist=2.6
- **passivity_window** r1.slp:3612 — passive_run=340f mid_dist=10.0
- **passivity_window** r1.slp:4754 — passive_run=320f mid_dist=26.3
- **passivity_window** r1.slp:6855 — passive_run=304f mid_dist=6.6
- **dropped_punish** r1.slp:533 — opening@533 start_pct=71.2 window=120f
- **dropped_punish** r1.slp:1062 — opening@1062 start_pct=83.2 window=120f
- **dropped_punish** r1.slp:1420 — opening@1420 start_pct=83.2 window=120f
- **dropped_punish** r1.slp:3479 — opening@3479 start_pct=211.0 window=120f
- **dropped_punish** r1.slp:6726 — opening@6726 start_pct=152.2 window=120f
- **neutral_loss** r1.slp:347 — hit@437 opener=p2_action44 dist_90f_prior=6.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1349 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
