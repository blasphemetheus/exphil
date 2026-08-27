# Coach Report — 20260827_182848

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.47 |
| Approaches (total) | 7 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 4 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 1.47 | 7 | 3 | 9 | 4 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r2.slp:973 — opening@973 start_pct=54.4 window=120f
- **dropped_punish** r2.slp:4670 — opening@4670 start_pct=15.0 window=120f
- **dropped_punish** r2.slp:5979 — opening@5979 start_pct=60.4 window=120f
- **dropped_punish** r2.slp:6355 — opening@6355 start_pct=75.4 window=120f
- **neutral_loss** r2.slp:688 — hit@778 opener=p2_action44 dist_90f_prior=21.6
- **neutral_loss** r2.slp:1324 — hit@1414 opener=p2_action60 dist_90f_prior=27.1
- **neutral_loss** r2.slp:1777 — hit@1867 opener=p2_action44 dist_90f_prior=20.8
- **neutral_loss** r2.slp:2890 — hit@2980 opener=p2_action44 dist_90f_prior=34.6
- **neutral_loss** r2.slp:3227 — hit@3317 opener=p2_action44 dist_90f_prior=50.9
- **neutral_loss** r2.slp:4372 — hit@4462 opener=p2_action44 dist_90f_prior=6.6

Appended 10 new gap(s) to `scenarios/gaps.json` (644 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
