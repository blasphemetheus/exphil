# Coach Report — 20260829_120245

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 8 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 7 |
| Death sequences | 3 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r7.slp | 0.49 | 8 | 2 | 8 | 7 | 3 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r7.slp:927 — passive_run=326f mid_dist=6.2
- **passivity_window** r7.slp:2837 — passive_run=343f mid_dist=16.8
- **passivity_window** r7.slp:3319 — passive_run=404f mid_dist=11.4
- **dropped_punish** r7.slp:920 — opening@920 start_pct=4.0 window=120f
- **dropped_punish** r7.slp:2232 — opening@2232 start_pct=40.2 window=120f
- **dropped_punish** r7.slp:3686 — opening@3686 start_pct=93.1 window=120f
- **dropped_punish** r7.slp:4240 — opening@4240 start_pct=131.1 window=120f
- **dropped_punish** r7.slp:5536 — opening@5536 start_pct=18.2 window=120f
- **dropped_punish** r7.slp:5874 — opening@5874 start_pct=40.0 window=120f
- **dropped_punish** r7.slp:6325 — opening@6325 start_pct=45.0 window=120f

Appended 10 new gap(s) to `scenarios/gaps.json` (1329 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
