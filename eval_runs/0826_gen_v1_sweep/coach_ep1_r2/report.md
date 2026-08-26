# Coach Report — 20260826_123015

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 2 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 2 | 0 | 10 | 2 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:3290 — passive_run=484f mid_dist=6.9
- **passivity_window** r2.slp:5657 — passive_run=697f mid_dist=21.1
- **dropped_punish** r2.slp:6641 — opening@6641 start_pct=365.9 window=120f
- **dropped_punish** r2.slp:7115 — opening@7115 start_pct=29.1 window=120f
- **neutral_loss** r2.slp:394 — hit@484 opener=p2_action44 dist_90f_prior=62.6
- **neutral_loss** r2.slp:1560 — hit@1650 opener=p2_action44 dist_90f_prior=13.9
- **neutral_loss** r2.slp:1984 — hit@2074 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r2.slp:2596 — hit@2686 opener=p2_action45 dist_90f_prior=19.8
- **neutral_loss** r2.slp:3183 — hit@3273 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r2.slp:3616 — hit@3706 opener=p2_action44 dist_90f_prior=6.9

Appended 10 new gap(s) to `scenarios/gaps.json` (246 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
