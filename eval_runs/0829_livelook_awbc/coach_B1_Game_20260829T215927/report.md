# Coach Report — 20260829_224649

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.9 |
| Approaches (total) | 8 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 1 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T215927.slp | 0.9 | 8 | 2 | 8 | 1 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T215927.slp:815 — opening@815 start_pct=19.5 window=120f
- **neutral_loss** Game_20260829T215927.slp:407 — hit@497 opener=p2_action360 dist_90f_prior=27.1
- **neutral_loss** Game_20260829T215927.slp:804 — hit@894 opener=p2_action69 dist_90f_prior=24.6
- **neutral_loss** Game_20260829T215927.slp:1282 — hit@1372 opener=p2_action50 dist_90f_prior=22.3
- **neutral_loss** Game_20260829T215927.slp:1587 — hit@1677 opener=p2_action56 dist_90f_prior=27.9
- **neutral_loss** Game_20260829T215927.slp:2041 — hit@2131 opener=p2_action67 dist_90f_prior=10.9
- **neutral_loss** Game_20260829T215927.slp:2399 — hit@2489 opener=p2_action68 dist_90f_prior=59.7
- **neutral_loss** Game_20260829T215927.slp:2881 — hit@2971 opener=p2_action65 dist_90f_prior=20.5
- **neutral_loss** Game_20260829T215927.slp:3538 — hit@3628 opener=p2_action68 dist_90f_prior=16.0
- **death_sequence** Game_20260829T215927.slp:600 — death@601 elapsed=1f opener=p2_action28

Appended 10 new gap(s) to `scenarios/gaps.json` (1658 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
