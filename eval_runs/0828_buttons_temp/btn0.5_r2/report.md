# Coach Report — 20260828_125918

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 2 |
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
| r2.slp | 0.49 | 3 | 2 | 9 | 4 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:2120 — passive_run=353f mid_dist=8.3
- **passivity_window** r2.slp:2884 — passive_run=313f mid_dist=8.3
- **dropped_punish** r2.slp:1797 — opening@1797 start_pct=100.7 window=120f
- **dropped_punish** r2.slp:3906 — opening@3906 start_pct=184.8 window=120f
- **dropped_punish** r2.slp:4532 — opening@4532 start_pct=7.7 window=120f
- **dropped_punish** r2.slp:6263 — opening@6263 start_pct=55.3 window=120f
- **neutral_loss** r2.slp:959 — hit@1049 opener=p2_action44 dist_90f_prior=24.9
- **neutral_loss** r2.slp:1337 — hit@1427 opener=p2_action257 dist_90f_prior=70.1
- **neutral_loss** r2.slp:2460 — hit@2550 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r2.slp:2767 — hit@2857 opener=p2_action44 dist_90f_prior=51.4

Appended 10 new gap(s) to `scenarios/gaps.json` (841 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
