# Coach Report — 20260829_231243

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.56 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 3 |
| Death sequences | 4 |
| Passivity windows | 5 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.56 | 2 | 1 | 7 | 3 | 4 | 5 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:666 — passive_run=392f mid_dist=7.0
- **passivity_window** r1.slp:1367 — passive_run=312f mid_dist=27.8
- **passivity_window** r1.slp:3286 — passive_run=375f mid_dist=8.3
- **passivity_window** r1.slp:3868 — passive_run=420f mid_dist=8.3
- **passivity_window** r1.slp:4480 — passive_run=477f mid_dist=1.3
- **dropped_punish** r1.slp:1163 — opening@1163 start_pct=106.2 window=120f
- **dropped_punish** r1.slp:4426 — opening@4426 start_pct=234.1 window=120f
- **dropped_punish** r1.slp:5167 — opening@5167 start_pct=9.8 window=120f
- **neutral_loss** r1.slp:395 — hit@485 opener=p2_action44 dist_90f_prior=5.1
- **neutral_loss** r1.slp:822 — hit@912 opener=p2_action44 dist_90f_prior=7.0

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
