# Coach Report — 20260829_231249

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 7 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 7 |
| Death sequences | 3 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.98 | 7 | 3 | 7 | 7 | 3 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:2235 — passive_run=323f mid_dist=8.3
- **passivity_window** r5.slp:6719 — passive_run=394f mid_dist=8.3
- **dropped_punish** r5.slp:369 — opening@369 start_pct=9.5 window=120f
- **dropped_punish** r5.slp:1824 — opening@1824 start_pct=80.1 window=120f
- **dropped_punish** r5.slp:3728 — opening@3728 start_pct=149.3 window=120f
- **dropped_punish** r5.slp:4735 — opening@4735 start_pct=166.0 window=120f
- **dropped_punish** r5.slp:5106 — opening@5106 start_pct=170.6 window=120f
- **dropped_punish** r5.slp:6045 — opening@6045 start_pct=200.7 window=120f
- **dropped_punish** r5.slp:6502 — opening@6502 start_pct=205.7 window=120f
- **neutral_loss** r5.slp:445 — hit@535 opener=p2_action44 dist_90f_prior=28.0

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
