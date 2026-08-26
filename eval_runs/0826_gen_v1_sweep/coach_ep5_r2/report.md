# Coach Report — 20260826_130211

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 3 | 0 | 5 | 3 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:3289 — passive_run=379f mid_dist=8.3
- **passivity_window** r2.slp:3907 — passive_run=335f mid_dist=6.6
- **dropped_punish** r2.slp:373 — opening@373 start_pct=18.2 window=120f
- **dropped_punish** r2.slp:2187 — opening@2187 start_pct=105.6 window=120f
- **dropped_punish** r2.slp:6406 — opening@6406 start_pct=93.1 window=120f
- **neutral_loss** r2.slp:1463 — hit@1553 opener=p2_action44 dist_90f_prior=24.0
- **neutral_loss** r2.slp:2375 — hit@2465 opener=p2_action44 dist_90f_prior=19.1
- **neutral_loss** r2.slp:3171 — hit@3261 opener=p2_action44 dist_90f_prior=20.2
- **neutral_loss** r2.slp:4057 — hit@4147 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r2.slp:4572 — hit@4662 opener=p2_action60 dist_90f_prior=19.4

Appended 10 new gap(s) to `scenarios/gaps.json` (365 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
