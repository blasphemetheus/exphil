# Coach Report — 20260827_181713

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 2 |
| Dropped punishes | 4 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.0 | 5 | 1 | 2 | 4 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r4.slp:708 — opening@708 start_pct=38.9 window=120f
- **dropped_punish** r4.slp:1401 — opening@1401 start_pct=48.9 window=120f
- **dropped_punish** r4.slp:1940 — opening@1940 start_pct=60.9 window=120f
- **dropped_punish** r4.slp:3399 — opening@3399 start_pct=102.7 window=120f
- **neutral_loss** r4.slp:2391 — hit@2481 opener=p2_action44 dist_90f_prior=20.7
- **neutral_loss** r4.slp:2998 — hit@3088 opener=p2_action60 dist_90f_prior=57.2
- **death_sequence** r4.slp:972 — death@973 elapsed=1f opener=p2_action16
- **death_sequence** r4.slp:1750 — death@1751 elapsed=1f opener=p2_action15
- **death_sequence** r4.slp:3533 — death@3534 elapsed=1f opener=p2_action14
- **death_sequence** r4.slp:3843 — death@3844 elapsed=1f opener=p2_action14

Appended 10 new gap(s) to `scenarios/gaps.json` (614 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
