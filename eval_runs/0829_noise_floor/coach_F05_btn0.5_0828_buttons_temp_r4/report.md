# Coach Report — 20260829_231233

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 4 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 7 |
| Death sequences | 2 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.49 | 4 | 2 | 5 | 7 | 2 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:2604 — passive_run=316f mid_dist=8.3
- **passivity_window** r4.slp:3032 — passive_run=333f mid_dist=8.3
- **passivity_window** r4.slp:3698 — passive_run=628f mid_dist=8.3
- **passivity_window** r4.slp:5602 — passive_run=383f mid_dist=0.1
- **dropped_punish** r4.slp:817 — opening@817 start_pct=75.5 window=120f
- **dropped_punish** r4.slp:1622 — opening@1622 start_pct=117.3 window=120f
- **dropped_punish** r4.slp:2313 — opening@2313 start_pct=129.4 window=120f
- **dropped_punish** r4.slp:3550 — opening@3550 start_pct=178.0 window=120f
- **dropped_punish** r4.slp:4528 — opening@4528 start_pct=223.5 window=120f
- **dropped_punish** r4.slp:5516 — opening@5516 start_pct=243.1 window=120f

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
