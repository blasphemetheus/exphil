# Coach Report — 20260829_231241

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 2 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.0 | 5 | 1 | 11 | 2 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r8.slp:2085 — opening@2085 start_pct=155.8 window=120f
- **dropped_punish** r8.slp:6688 — opening@6688 start_pct=37.1 window=120f
- **neutral_loss** r8.slp:771 — hit@861 opener=p2_action44 dist_90f_prior=16.6
- **neutral_loss** r8.slp:1022 — hit@1112 opener=p2_action44 dist_90f_prior=21.5
- **neutral_loss** r8.slp:1515 — hit@1605 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r8.slp:2266 — hit@2356 opener=p2_action356 dist_90f_prior=65.4
- **neutral_loss** r8.slp:2734 — hit@2824 opener=p2_action60 dist_90f_prior=6.6
- **neutral_loss** r8.slp:3266 — hit@3356 opener=p2_action44 dist_90f_prior=45.0
- **neutral_loss** r8.slp:3553 — hit@3643 opener=p2_action356 dist_90f_prior=80.7
- **neutral_loss** r8.slp:3962 — hit@4052 opener=p2_action44 dist_90f_prior=88.6

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
