# Coach Report — 20260828_162803

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 0 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 3 |
| Death sequences | 0 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 0 | 0 | 7 | 3 | 0 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:1413 — passive_run=407f mid_dist=8.3
- **passivity_window** r3.slp:3445 — passive_run=506f mid_dist=8.3
- **passivity_window** r3.slp:4642 — passive_run=530f mid_dist=8.3
- **passivity_window** r3.slp:5230 — passive_run=1011f mid_dist=8.3
- **dropped_punish** r3.slp:2024 — opening@2024 start_pct=186.1 window=120f
- **dropped_punish** r3.slp:2378 — opening@2378 start_pct=204.2 window=120f
- **dropped_punish** r3.slp:6882 — opening@6882 start_pct=12.9 window=120f
- **neutral_loss** r3.slp:933 — hit@1023 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r3.slp:1266 — hit@1356 opener=p2_action44 dist_90f_prior=8.7
- **neutral_loss** r3.slp:1849 — hit@1939 opener=p2_action57 dist_90f_prior=56.2

Appended 10 new gap(s) to `scenarios/gaps.json` (901 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, passivity.
