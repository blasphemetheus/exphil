# Coach Report — 20260828_171152

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 3.32 |
| Approaches (total) | 11 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T053835.slp | 3.32 | 11 | 1 | 5 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T053835.slp:2899 — opening@2899 start_pct=98.0 window=120f
- **neutral_loss** Game_20260828T053835.slp:565 — hit@655 opener=p2_action215 dist_90f_prior=34.3
- **neutral_loss** Game_20260828T053835.slp:2264 — hit@2354 opener=p2_action56 dist_90f_prior=27.4
- **neutral_loss** Game_20260828T053835.slp:2809 — hit@2899 opener=p2_action85 dist_90f_prior=26.0
- **neutral_loss** Game_20260828T053835.slp:3134 — hit@3224 opener=p2_action360 dist_90f_prior=73.4
- **neutral_loss** Game_20260828T053835.slp:3984 — hit@4074 opener=p2_action213 dist_90f_prior=32.7
- **death_sequence** Game_20260828T053835.slp:357 — death@358 elapsed=1f opener=p2_action14
- **death_sequence** Game_20260828T053835.slp:2078 — death@2079 elapsed=1f opener=p2_action25
- **death_sequence** Game_20260828T053835.slp:3375 — death@3376 elapsed=1f opener=p2_action42
- **death_sequence** Game_20260828T053835.slp:4340 — death@4341 elapsed=1f opener=p2_action27

Appended 10 new gap(s) to `scenarios/gaps.json` (1190 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
