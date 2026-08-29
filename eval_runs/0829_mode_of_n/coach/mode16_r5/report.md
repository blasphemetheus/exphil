# Coach Report — 20260829_150201

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 1 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.0 | 3 | 0 | 1 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r5.slp:793 — opening@793 start_pct=13.7 window=120f
- **neutral_loss** r5.slp:436 — hit@526 opener=p2_action57 dist_90f_prior=11.3
- **death_sequence** r5.slp:917 — death@918 elapsed=1f opener=p2_action18
- **death_sequence** r5.slp:1180 — death@1181 elapsed=1f opener=p2_action16
- **death_sequence** r5.slp:1441 — death@1442 elapsed=1f opener=p2_action14
- **death_sequence** r5.slp:1691 — death@1692 elapsed=1f opener=p2_action14

Appended 6 new gap(s) to `scenarios/gaps.json` (1609 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
