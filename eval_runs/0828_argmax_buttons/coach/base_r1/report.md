# Coach Report — 20260828_162801

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 6 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.98 | 4 | 1 | 9 | 6 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:3319 — passive_run=301f mid_dist=8.3
- **passivity_window** r1.slp:3992 — passive_run=316f mid_dist=6.9
- **dropped_punish** r1.slp:774 — opening@774 start_pct=44.3 window=120f
- **dropped_punish** r1.slp:1341 — opening@1341 start_pct=59.0 window=120f
- **dropped_punish** r1.slp:1682 — opening@1682 start_pct=72.6 window=120f
- **dropped_punish** r1.slp:5903 — opening@5903 start_pct=280.4 window=120f
- **dropped_punish** r1.slp:6574 — opening@6574 start_pct=9.5 window=120f
- **dropped_punish** r1.slp:7075 — opening@7075 start_pct=24.8 window=120f
- **neutral_loss** r1.slp:1046 — hit@1136 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r1.slp:2228 — hit@2318 opener=p2_action44 dist_90f_prior=20.6

Appended 10 new gap(s) to `scenarios/gaps.json` (881 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
