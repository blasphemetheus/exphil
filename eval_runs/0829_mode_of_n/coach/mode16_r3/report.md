# Coach Report — 20260829_150159

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 3 | 1 | 8 | 0 | 4 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:871 — passive_run=304f mid_dist=7.5
- **passivity_window** r3.slp:2501 — passive_run=412f mid_dist=25.1
- **passivity_window** r3.slp:2982 — passive_run=360f mid_dist=27.0
- **passivity_window** r3.slp:3348 — passive_run=367f mid_dist=23.2
- **neutral_loss** r3.slp:348 — hit@438 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:685 — hit@775 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:1055 — hit@1145 opener=p2_action44 dist_90f_prior=7.5
- **neutral_loss** r3.slp:1394 — hit@1484 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r3.slp:1788 — hit@1878 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:2119 — hit@2209 opener=p2_action44 dist_90f_prior=52.3

Appended 10 new gap(s) to `scenarios/gaps.json` (1595 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
