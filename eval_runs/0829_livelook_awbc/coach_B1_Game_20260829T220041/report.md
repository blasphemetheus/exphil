# Coach Report — 20260829_224650

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.26 |
| Approaches (total) | 1 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 3 |
| Dropped punishes | 2 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T220041.slp | 1.26 | 1 | 1 | 3 | 2 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T220041.slp:1829 — opening@1829 start_pct=24.0 window=120f
- **dropped_punish** Game_20260829T220041.slp:2462 — opening@2462 start_pct=25.0 window=120f
- **neutral_loss** Game_20260829T220041.slp:1064 — hit@1154 opener=p2_action63 dist_90f_prior=22.0
- **neutral_loss** Game_20260829T220041.slp:1836 — hit@1926 opener=p2_action69 dist_90f_prior=9.7
- **neutral_loss** Game_20260829T220041.slp:2372 — hit@2462 opener=p2_action85 dist_90f_prior=45.9
- **death_sequence** Game_20260829T220041.slp:726 — death@727 elapsed=1f opener=p2_action253
- **death_sequence** Game_20260829T220041.slp:1526 — death@1527 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260829T220041.slp:2190 — death@2191 elapsed=1f opener=p2_action236
- **death_sequence** Game_20260829T220041.slp:2849 — death@2850 elapsed=1f opener=p2_action25

Appended 9 new gap(s) to `scenarios/gaps.json` (1667 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
