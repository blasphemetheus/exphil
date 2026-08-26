# Coach Report — 20260826_123832

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 5 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.98 | 5 | 2 | 7 | 5 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:2191 — passive_run=387f mid_dist=7.1
- **passivity_window** r3.slp:5077 — passive_run=375f mid_dist=8.3
- **dropped_punish** r3.slp:355 — opening@355 start_pct=8.0 window=120f
- **dropped_punish** r3.slp:698 — opening@698 start_pct=27.7 window=120f
- **dropped_punish** r3.slp:3068 — opening@3068 start_pct=142.6 window=120f
- **dropped_punish** r3.slp:3480 — opening@3480 start_pct=149.6 window=120f
- **dropped_punish** r3.slp:3832 — opening@3832 start_pct=154.6 window=120f
- **neutral_loss** r3.slp:1925 — hit@2015 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:2464 — hit@2554 opener=p2_action60 dist_90f_prior=20.0
- **neutral_loss** r3.slp:3636 — hit@3726 opener=p2_action60 dist_90f_prior=53.1

Appended 10 new gap(s) to `scenarios/gaps.json` (286 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
