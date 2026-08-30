# Coach Report — 20260829_224717

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 3.02 |
| Approaches (total) | 7 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 3 |
| Dropped punishes | 0 |
| Death sequences | 3 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T221610.slp | 3.02 | 7 | 1 | 3 | 0 | 3 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T221610.slp:382 — passive_run=339f mid_dist=9.4
- **passivity_window** Game_20260829T221610.slp:1788 — passive_run=395f mid_dist=21.2
- **neutral_loss** Game_20260829T221610.slp:310 — hit@400 opener=p2_action360 dist_90f_prior=38.1
- **neutral_loss** Game_20260829T221610.slp:1951 — hit@2041 opener=p2_action50 dist_90f_prior=19.8
- **neutral_loss** Game_20260829T221610.slp:2917 — hit@3007 opener=p2_action215 dist_90f_prior=25.6
- **death_sequence** Game_20260829T221610.slp:981 — death@982 elapsed=1f opener=p2_action25
- **death_sequence** Game_20260829T221610.slp:2467 — death@2468 elapsed=1f opener=p2_action43
- **death_sequence** Game_20260829T221610.slp:3572 — death@3573 elapsed=1f opener=p2_action344

Appended 8 new gap(s) to `scenarios/gaps.json` (1820 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
