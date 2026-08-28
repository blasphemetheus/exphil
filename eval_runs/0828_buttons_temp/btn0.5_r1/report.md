# Coach Report — 20260828_125917

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.47 |
| Approaches (total) | 11 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 1.47 | 11 | 2 | 9 | 5 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r1.slp:2053 — opening@2053 start_pct=107.3 window=120f
- **dropped_punish** r1.slp:3243 — opening@3243 start_pct=144.3 window=120f
- **dropped_punish** r1.slp:4490 — opening@4490 start_pct=18.1 window=120f
- **dropped_punish** r1.slp:5910 — opening@5910 start_pct=9.0 window=120f
- **dropped_punish** r1.slp:6261 — opening@6261 start_pct=28.5 window=120f
- **neutral_loss** r1.slp:951 — hit@1041 opener=p2_action45 dist_90f_prior=15.8
- **neutral_loss** r1.slp:1312 — hit@1402 opener=p2_action48 dist_90f_prior=20.5
- **neutral_loss** r1.slp:1746 — hit@1836 opener=p2_action57 dist_90f_prior=34.1
- **neutral_loss** r1.slp:2082 — hit@2172 opener=p2_action256 dist_90f_prior=36.6
- **neutral_loss** r1.slp:2391 — hit@2481 opener=p2_action44 dist_90f_prior=70.7

Appended 10 new gap(s) to `scenarios/gaps.json` (831 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
