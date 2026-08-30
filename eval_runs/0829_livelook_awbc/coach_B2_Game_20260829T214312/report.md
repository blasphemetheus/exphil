# Coach Report — 20260829_224700

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T214312.slp | 0.0 | 5 | 1 | 5 | 0 | 4 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T214312.slp:483 — passive_run=328f mid_dist=25.1
- **passivity_window** Game_20260829T214312.slp:2591 — passive_run=315f mid_dist=9.5
- **passivity_window** Game_20260829T214312.slp:3197 — passive_run=387f mid_dist=33.6
- **neutral_loss** Game_20260829T214312.slp:411 — hit@501 opener=p2_action215 dist_90f_prior=43.8
- **neutral_loss** Game_20260829T214312.slp:1267 — hit@1357 opener=p2_action215 dist_90f_prior=44.9
- **neutral_loss** Game_20260829T214312.slp:2249 — hit@2339 opener=p2_action68 dist_90f_prior=32.0
- **neutral_loss** Game_20260829T214312.slp:2571 — hit@2661 opener=p2_action215 dist_90f_prior=55.2
- **neutral_loss** Game_20260829T214312.slp:3206 — hit@3296 opener=p2_action67 dist_90f_prior=33.4
- **death_sequence** Game_20260829T214312.slp:886 — death@887 elapsed=1f opener=p2_action0
- **death_sequence** Game_20260829T214312.slp:1805 — death@1806 elapsed=1f opener=p2_action28

Appended 10 new gap(s) to `scenarios/gaps.json` (1720 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
