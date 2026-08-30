# Coach Report — 20260829_224708

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.7 |
| Approaches (total) | 7 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T220638.slp | 1.7 | 7 | 1 | 6 | 0 | 4 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T220638.slp:3935 — passive_run=424f mid_dist=20.8
- **neutral_loss** Game_20260829T220638.slp:403 — hit@493 opener=p2_action56 dist_90f_prior=21.8
- **neutral_loss** Game_20260829T220638.slp:1765 — hit@1855 opener=p2_action67 dist_90f_prior=32.5
- **neutral_loss** Game_20260829T220638.slp:4672 — hit@4762 opener=p2_action67 dist_90f_prior=50.8
- **neutral_loss** Game_20260829T220638.slp:5112 — hit@5202 opener=p2_action63 dist_90f_prior=17.2
- **neutral_loss** Game_20260829T220638.slp:5352 — hit@5442 opener=p2_action63 dist_90f_prior=8.3
- **neutral_loss** Game_20260829T220638.slp:5631 — hit@5721 opener=p2_action67 dist_90f_prior=3.6
- **death_sequence** Game_20260829T220638.slp:1145 — death@1146 elapsed=1f opener=p2_action253
- **death_sequence** Game_20260829T220638.slp:2576 — death@2577 elapsed=1f opener=p2_action25
- **death_sequence** Game_20260829T220638.slp:4840 — death@4841 elapsed=1f opener=p2_action29

Appended 10 new gap(s) to `scenarios/gaps.json` (1773 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
