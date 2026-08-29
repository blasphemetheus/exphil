# Coach Report — 20260829_120255

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 7 |
| Death sequences | 1 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.98 | 4 | 1 | 6 | 7 | 1 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r2.slp:2353 — opening@2353 start_pct=129.5 window=120f
- **dropped_punish** r2.slp:2751 — opening@2751 start_pct=142.0 window=120f
- **dropped_punish** r2.slp:3257 — opening@3257 start_pct=147.0 window=120f
- **dropped_punish** r2.slp:4571 — opening@4571 start_pct=205.0 window=120f
- **dropped_punish** r2.slp:5306 — opening@5306 start_pct=236.9 window=120f
- **dropped_punish** r2.slp:6069 — opening@6069 start_pct=254.4 window=120f
- **dropped_punish** r2.slp:7033 — opening@7033 start_pct=35.7 window=120f
- **neutral_loss** r2.slp:399 — hit@489 opener=p2_action44 dist_90f_prior=8.7
- **neutral_loss** r2.slp:1403 — hit@1493 opener=p2_action53 dist_90f_prior=35.0
- **neutral_loss** r2.slp:1676 — hit@1766 opener=p2_action44 dist_90f_prior=6.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1439 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
