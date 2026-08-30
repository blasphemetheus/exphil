# Coach Report — 20260829_224710

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.71 |
| Approaches (total) | 6 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 4 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T220831.slp | 0.71 | 6 | 1 | 4 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T220831.slp:4021 — opening@4021 start_pct=117.5 window=120f
- **neutral_loss** Game_20260829T220831.slp:432 — hit@522 opener=p2_action360 dist_90f_prior=8.3
- **neutral_loss** Game_20260829T220831.slp:975 — hit@1065 opener=p2_action213 dist_90f_prior=9.4
- **neutral_loss** Game_20260829T220831.slp:2635 — hit@2725 opener=p2_action68 dist_90f_prior=33.4
- **neutral_loss** Game_20260829T220831.slp:3393 — hit@3483 opener=p2_action213 dist_90f_prior=4.3
- **death_sequence** Game_20260829T220831.slp:647 — death@648 elapsed=1f opener=p2_action20
- **death_sequence** Game_20260829T220831.slp:2928 — death@2929 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260829T220831.slp:4578 — death@4579 elapsed=1f opener=p2_action50
- **death_sequence** Game_20260829T220831.slp:5033 — death@5034 elapsed=1f opener=p2_action20

Appended 9 new gap(s) to `scenarios/gaps.json` (1782 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
