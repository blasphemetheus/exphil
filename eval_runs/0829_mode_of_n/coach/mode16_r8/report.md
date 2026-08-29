# Coach Report — 20260829_150203

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.0 | 4 | 0 | 5 | 0 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **neutral_loss** r8.slp:486 — hit@576 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r8.slp:838 — hit@928 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r8.slp:1193 — hit@1283 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r8.slp:1565 — hit@1655 opener=p2_action44 dist_90f_prior=19.0
- **neutral_loss** r8.slp:1894 — hit@1984 opener=p2_action44 dist_90f_prior=7.0
- **death_sequence** r8.slp:2261 — death@2262 elapsed=1f opener=p2_action14
- **death_sequence** r8.slp:2522 — death@2523 elapsed=1f opener=p2_action15
- **death_sequence** r8.slp:2783 — death@2784 elapsed=1f opener=p2_action14
- **death_sequence** r8.slp:3044 — death@3045 elapsed=1f opener=p2_action14

Appended 9 new gap(s) to `scenarios/gaps.json` (1638 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths.
