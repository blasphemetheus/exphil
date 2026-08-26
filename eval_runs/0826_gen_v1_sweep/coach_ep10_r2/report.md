# Coach Report — 20260826_134317

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 5 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 6 |
| Death sequences | 3 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.49 | 5 | 3 | 11 | 6 | 3 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:2658 — passive_run=313f mid_dist=17.0
- **passivity_window** r2.slp:6496 — passive_run=332f mid_dist=6.9
- **dropped_punish** r2.slp:1056 — opening@1056 start_pct=61.5 window=120f
- **dropped_punish** r2.slp:1464 — opening@1464 start_pct=69.5 window=120f
- **dropped_punish** r2.slp:2658 — opening@2658 start_pct=128.9 window=120f
- **dropped_punish** r2.slp:4315 — opening@4315 start_pct=197.0 window=120f
- **dropped_punish** r2.slp:5277 — opening@5277 start_pct=229.5 window=120f
- **dropped_punish** r2.slp:7018 — opening@7018 start_pct=73.1 window=120f
- **neutral_loss** r2.slp:638 — hit@728 opener=p2_action44 dist_90f_prior=16.0
- **neutral_loss** r2.slp:1200 — hit@1290 opener=p2_action57 dist_90f_prior=44.3

Appended 10 new gap(s) to `scenarios/gaps.json` (515 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
