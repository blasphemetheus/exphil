# Coach Report — 20260829_224703

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 11 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 2 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T214624.slp | 0.0 | 11 | 1 | 2 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T214624.slp:756 — opening@756 start_pct=15.0 window=120f
- **neutral_loss** Game_20260829T214624.slp:1989 — hit@2079 opener=p2_action63 dist_90f_prior=12.4
- **neutral_loss** Game_20260829T214624.slp:2742 — hit@2832 opener=p2_action215 dist_90f_prior=25.8
- **death_sequence** Game_20260829T214624.slp:521 — death@522 elapsed=1f opener=p2_action24
- **death_sequence** Game_20260829T214624.slp:829 — death@830 elapsed=1f opener=p2_action42
- **death_sequence** Game_20260829T214624.slp:2421 — death@2422 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260829T214624.slp:3486 — death@3487 elapsed=1f opener=p2_action365

Appended 7 new gap(s) to `scenarios/gaps.json` (1737 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
