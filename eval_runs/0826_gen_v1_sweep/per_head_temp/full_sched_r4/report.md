# Coach Report — 20260827_182850

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.52 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 7 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.52 | 5 | 1 | 7 | 7 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r4.slp:1332 — opening@1332 start_pct=75.7 window=120f
- **dropped_punish** r4.slp:2140 — opening@2140 start_pct=84.7 window=120f
- **dropped_punish** r4.slp:2506 — opening@2506 start_pct=95.4 window=120f
- **dropped_punish** r4.slp:2962 — opening@2962 start_pct=116.0 window=120f
- **dropped_punish** r4.slp:4313 — opening@4313 start_pct=165.2 window=120f
- **dropped_punish** r4.slp:4829 — opening@4829 start_pct=4.3 window=120f
- **dropped_punish** r4.slp:5678 — opening@5678 start_pct=92.3 window=120f
- **neutral_loss** r4.slp:311 — hit@401 opener=p2_action44 dist_90f_prior=59.3
- **neutral_loss** r4.slp:671 — hit@761 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r4.slp:2308 — hit@2398 opener=p2_action44 dist_90f_prior=18.6

Appended 10 new gap(s) to `scenarios/gaps.json` (664 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
