# Coach Report — 20260826_122125

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 2 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 3 | 0 | 2 | 0 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **neutral_loss** r1.slp:364 — hit@454 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r1.slp:795 — hit@885 opener=p2_action44 dist_90f_prior=11.7
- **death_sequence** r1.slp:932 — death@1025 elapsed=93f opener=p2_action44
- **death_sequence** r1.slp:1285 — death@1286 elapsed=1f opener=p2_action14
- **death_sequence** r1.slp:1546 — death@1547 elapsed=1f opener=p2_action14
- **death_sequence** r1.slp:1807 — death@1808 elapsed=1f opener=p2_action14

Appended 6 new gap(s) to `scenarios/gaps.json` (206 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths.
