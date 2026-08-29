# Coach Report — 20260829_150200

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 3 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.0 | 3 | 0 | 3 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r4.slp:697 — opening@697 start_pct=15.5 window=120f
- **neutral_loss** r4.slp:356 — hit@446 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r4.slp:777 — hit@867 opener=p2_action63 dist_90f_prior=57.8
- **neutral_loss** r4.slp:1065 — hit@1155 opener=p2_action44 dist_90f_prior=5.9
- **death_sequence** r4.slp:1244 — death@1245 elapsed=1f opener=p2_action14
- **death_sequence** r4.slp:1494 — death@1495 elapsed=1f opener=p2_action14
- **death_sequence** r4.slp:1755 — death@1756 elapsed=1f opener=p2_action15
- **death_sequence** r4.slp:2016 — death@2017 elapsed=1f opener=p2_action16

Appended 8 new gap(s) to `scenarios/gaps.json` (1603 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
