# Coach Report — 20260826_131841

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 1 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.98 | 2 | 1 | 11 | 1 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:5561 — passive_run=316f mid_dist=6.8
- **passivity_window** r2.slp:6823 — passive_run=346f mid_dist=6.6
- **dropped_punish** r2.slp:4116 — opening@4116 start_pct=235.7 window=120f
- **neutral_loss** r2.slp:305 — hit@395 opener=p2_action44 dist_90f_prior=57.5
- **neutral_loss** r2.slp:690 — hit@780 opener=p2_action44 dist_90f_prior=19.4
- **neutral_loss** r2.slp:1551 — hit@1641 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r2.slp:2767 — hit@2857 opener=p2_action60 dist_90f_prior=22.3
- **neutral_loss** r2.slp:3026 — hit@3116 opener=p2_action44 dist_90f_prior=73.7
- **neutral_loss** r2.slp:3362 — hit@3452 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r2.slp:4872 — hit@4962 opener=p2_action44 dist_90f_prior=8.7

Appended 10 new gap(s) to `scenarios/gaps.json` (425 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
