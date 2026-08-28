# Coach Report — 20260828_171127

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.99 |
| Approaches (total) | 14 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T165031.slp | 0.99 | 14 | 0 | 7 | 0 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **neutral_loss** Game_20260828T165031.slp:386 — hit@476 opener=p2_action68 dist_90f_prior=38.7
- **neutral_loss** Game_20260828T165031.slp:1110 — hit@1200 opener=p2_action65 dist_90f_prior=43.9
- **neutral_loss** Game_20260828T165031.slp:1572 — hit@1662 opener=p2_action356 dist_90f_prior=39.5
- **neutral_loss** Game_20260828T165031.slp:4272 — hit@4362 opener=p2_action360 dist_90f_prior=8.6
- **neutral_loss** Game_20260828T165031.slp:4541 — hit@4631 opener=p2_action352 dist_90f_prior=38.7
- **neutral_loss** Game_20260828T165031.slp:5346 — hit@5436 opener=p2_action69 dist_90f_prior=21.0
- **neutral_loss** Game_20260828T165031.slp:5982 — hit@6072 opener=p2_action65 dist_90f_prior=27.2
- **death_sequence** Game_20260828T165031.slp:1830 — death@1831 elapsed=1f opener=p2_action354
- **death_sequence** Game_20260828T165031.slp:2427 — death@2428 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260828T165031.slp:4109 — death@4110 elapsed=1f opener=p2_action27

Appended 10 new gap(s) to `scenarios/gaps.json` (1091 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths.
