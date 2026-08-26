# Coach Report — 20260826_132659

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 0 |
| Death sequences | 0 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 2 | 1 | 9 | 0 | 0 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:1952 — passive_run=376f mid_dist=21.9
- **passivity_window** r3.slp:3050 — passive_run=508f mid_dist=12.4
- **passivity_window** r3.slp:3638 — passive_run=471f mid_dist=8.3
- **neutral_loss** r3.slp:1496 — hit@1586 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:2022 — hit@2112 opener=p2_action53 dist_90f_prior=21.8
- **neutral_loss** r3.slp:2770 — hit@2860 opener=p2_action44 dist_90f_prior=35.4
- **neutral_loss** r3.slp:3460 — hit@3550 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r3.slp:4053 — hit@4143 opener=p2_action57 dist_90f_prior=17.3
- **neutral_loss** r3.slp:4302 — hit@4392 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r3.slp:4833 — hit@4923 opener=p2_action44 dist_90f_prior=7.0

Appended 10 new gap(s) to `scenarios/gaps.json` (465 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, passivity.
