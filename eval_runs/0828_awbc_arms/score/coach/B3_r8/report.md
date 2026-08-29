# Coach Report — 20260829_120300

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 7 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.98 | 7 | 1 | 8 | 5 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r8.slp:695 — passive_run=416f mid_dist=20.1
- **dropped_punish** r8.slp:350 — opening@350 start_pct=39.4 window=120f
- **dropped_punish** r8.slp:1476 — opening@1476 start_pct=82.0 window=120f
- **dropped_punish** r8.slp:1811 — opening@1811 start_pct=89.0 window=120f
- **dropped_punish** r8.slp:3002 — opening@3002 start_pct=104.9 window=120f
- **dropped_punish** r8.slp:4641 — opening@4641 start_pct=34.0 window=120f
- **neutral_loss** r8.slp:480 — hit@570 opener=p2_action60 dist_90f_prior=62.1
- **neutral_loss** r8.slp:767 — hit@857 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r8.slp:1958 — hit@2048 opener=p2_action44 dist_90f_prior=7.6
- **neutral_loss** r8.slp:3980 — hit@4070 opener=p2_action60 dist_90f_prior=7.4

Appended 10 new gap(s) to `scenarios/gaps.json` (1499 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
