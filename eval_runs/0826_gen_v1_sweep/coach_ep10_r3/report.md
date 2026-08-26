# Coach Report — 20260826_134318

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 4 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 3 | 3 | 9 | 4 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:1904 — passive_run=305f mid_dist=8.3
- **passivity_window** r3.slp:4990 — passive_run=685f mid_dist=9.7
- **dropped_punish** r3.slp:1163 — opening@1163 start_pct=73.5 window=120f
- **dropped_punish** r3.slp:4059 — opening@4059 start_pct=169.0 window=120f
- **dropped_punish** r3.slp:6346 — opening@6346 start_pct=254.8 window=120f
- **dropped_punish** r3.slp:7146 — opening@7146 start_pct=291.0 window=120f
- **neutral_loss** r3.slp:1194 — hit@1284 opener=p2_action44 dist_90f_prior=54.5
- **neutral_loss** r3.slp:1524 — hit@1614 opener=p2_action44 dist_90f_prior=22.1
- **neutral_loss** r3.slp:2967 — hit@3057 opener=p2_action44 dist_90f_prior=0.0
- **neutral_loss** r3.slp:4123 — hit@4213 opener=p2_action57 dist_90f_prior=56.8

Appended 10 new gap(s) to `scenarios/gaps.json` (525 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
