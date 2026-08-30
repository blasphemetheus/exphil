# Coach Report — 20260829_224714

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 12 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 2 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T221232.slp | 0.0 | 12 | 0 | 5 | 2 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T221232.slp:401 — opening@401 start_pct=15.8 window=120f
- **dropped_punish** Game_20260829T221232.slp:3247 — opening@3247 start_pct=53.3 window=120f
- **neutral_loss** Game_20260829T221232.slp:479 — hit@569 opener=p2_action69 dist_90f_prior=34.4
- **neutral_loss** Game_20260829T221232.slp:1845 — hit@1935 opener=p2_action215 dist_90f_prior=22.9
- **neutral_loss** Game_20260829T221232.slp:3488 — hit@3578 opener=p2_action63 dist_90f_prior=21.6
- **neutral_loss** Game_20260829T221232.slp:4145 — hit@4235 opener=p2_action50 dist_90f_prior=30.1
- **neutral_loss** Game_20260829T221232.slp:5034 — hit@5124 opener=p2_action67 dist_90f_prior=54.5
- **death_sequence** Game_20260829T221232.slp:1453 — death@1485 elapsed=32f opener=p2_action85
- **death_sequence** Game_20260829T221232.slp:3077 — death@3078 elapsed=1f opener=p2_action29
- **death_sequence** Game_20260829T221232.slp:4843 — death@4844 elapsed=1f opener=p2_action27

Appended 10 new gap(s) to `scenarios/gaps.json` (1802 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
