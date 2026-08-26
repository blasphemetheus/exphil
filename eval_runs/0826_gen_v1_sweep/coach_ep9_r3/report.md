# Coach Report — 20260826_133516

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 4 |
| Dropped punishes | 0 |
| Death sequences | 2 |
| Passivity windows | 8 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 2 | 0 | 4 | 0 | 2 | 8 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:2069 — passive_run=315f mid_dist=8.3
- **passivity_window** r3.slp:2465 — passive_run=340f mid_dist=8.3
- **passivity_window** r3.slp:3283 — passive_run=322f mid_dist=8.3
- **passivity_window** r3.slp:3666 — passive_run=327f mid_dist=8.3
- **passivity_window** r3.slp:3999 — passive_run=374f mid_dist=8.3
- **passivity_window** r3.slp:4379 — passive_run=421f mid_dist=8.3
- **passivity_window** r3.slp:5328 — passive_run=561f mid_dist=8.3
- **passivity_window** r3.slp:6449 — passive_run=481f mid_dist=8.3
- **neutral_loss** r3.slp:1044 — hit@1134 opener=p2_action60 dist_90f_prior=64.9
- **neutral_loss** r3.slp:3177 — hit@3267 opener=p2_action44 dist_90f_prior=36.1

Appended 10 new gap(s) to `scenarios/gaps.json` (495 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
