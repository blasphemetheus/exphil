# Coach Report — 20260829_231238

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 5 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 7 |
| Death sequences | 2 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 5 | 2 | 7 | 7 | 2 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:3722 — passive_run=430f mid_dist=6.9
- **passivity_window** r5.slp:5655 — passive_run=416f mid_dist=19.0
- **passivity_window** r5.slp:6620 — passive_run=305f mid_dist=8.3
- **dropped_punish** r5.slp:606 — opening@606 start_pct=64.2 window=120f
- **dropped_punish** r5.slp:982 — opening@982 start_pct=90.9 window=120f
- **dropped_punish** r5.slp:1937 — opening@1937 start_pct=138.4 window=120f
- **dropped_punish** r5.slp:3188 — opening@3188 start_pct=230.8 window=120f
- **dropped_punish** r5.slp:3531 — opening@3531 start_pct=237.8 window=120f
- **dropped_punish** r5.slp:4151 — opening@4151 start_pct=244.1 window=120f
- **dropped_punish** r5.slp:4415 — opening@4415 start_pct=259.1 window=120f

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
