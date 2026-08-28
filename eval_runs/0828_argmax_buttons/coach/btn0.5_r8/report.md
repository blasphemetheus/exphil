# Coach Report — 20260828_162814

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 4 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.98 | 4 | 2 | 10 | 4 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r8.slp:639 — passive_run=303f mid_dist=18.1
- **passivity_window** r8.slp:5663 — passive_run=356f mid_dist=0.7
- **dropped_punish** r8.slp:405 — opening@405 start_pct=24.4 window=120f
- **dropped_punish** r8.slp:923 — opening@923 start_pct=54.1 window=120f
- **dropped_punish** r8.slp:6349 — opening@6349 start_pct=120.4 window=120f
- **dropped_punish** r8.slp:6770 — opening@6770 start_pct=127.8 window=120f
- **neutral_loss** r8.slp:435 — hit@525 opener=p2_action53 dist_90f_prior=41.6
- **neutral_loss** r8.slp:978 — hit@1068 opener=p2_action60 dist_90f_prior=57.6
- **neutral_loss** r8.slp:2129 — hit@2219 opener=p2_action44 dist_90f_prior=45.4
- **neutral_loss** r8.slp:2485 — hit@2575 opener=p2_action44 dist_90f_prior=7.0

Appended 10 new gap(s) to `scenarios/gaps.json` (991 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
