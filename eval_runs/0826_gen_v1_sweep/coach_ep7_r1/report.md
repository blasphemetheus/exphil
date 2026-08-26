# Coach Report — 20260826_131840

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 6 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 4 | 1 | 12 | 6 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r1.slp:445 — opening@445 start_pct=43.5 window=120f
- **dropped_punish** r1.slp:1443 — opening@1443 start_pct=57.5 window=120f
- **dropped_punish** r1.slp:1945 — opening@1945 start_pct=57.5 window=120f
- **dropped_punish** r1.slp:3252 — opening@3252 start_pct=0.0 window=120f
- **dropped_punish** r1.slp:3980 — opening@3980 start_pct=27.5 window=120f
- **dropped_punish** r1.slp:4359 — opening@4359 start_pct=56.5 window=120f
- **neutral_loss** r1.slp:437 — hit@527 opener=p2_action60 dist_90f_prior=6.6
- **neutral_loss** r1.slp:915 — hit@1005 opener=p2_action60 dist_90f_prior=65.4
- **neutral_loss** r1.slp:1509 — hit@1599 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r1.slp:2275 — hit@2365 opener=p2_action44 dist_90f_prior=6.8

Appended 10 new gap(s) to `scenarios/gaps.json` (415 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
