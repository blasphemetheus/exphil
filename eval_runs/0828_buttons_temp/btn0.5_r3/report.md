# Coach Report — 20260828_125919

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 6 |
| Death sequences | 3 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 1 | 0 | 6 | 6 | 3 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:1800 — passive_run=360f mid_dist=15.5
- **passivity_window** r3.slp:3634 — passive_run=417f mid_dist=8.3
- **passivity_window** r3.slp:4861 — passive_run=450f mid_dist=8.3
- **dropped_punish** r3.slp:753 — opening@753 start_pct=64.7 window=120f
- **dropped_punish** r3.slp:1712 — opening@1712 start_pct=83.7 window=120f
- **dropped_punish** r3.slp:2165 — opening@2165 start_pct=105.1 window=120f
- **dropped_punish** r3.slp:3039 — opening@3039 start_pct=185.0 window=120f
- **dropped_punish** r3.slp:5614 — opening@5614 start_pct=263.8 window=120f
- **dropped_punish** r3.slp:5954 — opening@5954 start_pct=271.2 window=120f
- **neutral_loss** r3.slp:893 — hit@983 opener=p2_action44 dist_90f_prior=54.4

Appended 10 new gap(s) to `scenarios/gaps.json` (851 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
