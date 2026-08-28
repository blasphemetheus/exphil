# Coach Report — 20260828_123231

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 3 |
| Death sequences | 3 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 3 | 1 | 6 | 3 | 3 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:4983 — passive_run=348f mid_dist=26.5
- **dropped_punish** r5.slp:4172 — opening@4172 start_pct=178.9 window=120f
- **dropped_punish** r5.slp:6784 — opening@6784 start_pct=55.1 window=120f
- **dropped_punish** r5.slp:7126 — opening@7126 start_pct=59.2 window=120f
- **neutral_loss** r5.slp:1790 — hit@1880 opener=p2_action44 dist_90f_prior=18.3
- **neutral_loss** r5.slp:2299 — hit@2389 opener=p2_action44 dist_90f_prior=19.8
- **neutral_loss** r5.slp:4357 — hit@4447 opener=p2_action44 dist_90f_prior=7.8
- **neutral_loss** r5.slp:5202 — hit@5292 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r5.slp:6863 — hit@6953 opener=p2_action44 dist_90f_prior=46.4
- **neutral_loss** r5.slp:7198 — hit@7288 opener=p2_action44 dist_90f_prior=58.1

Appended 10 new gap(s) to `scenarios/gaps.json` (771 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
