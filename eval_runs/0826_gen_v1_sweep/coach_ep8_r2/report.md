# Coach Report — 20260826_132658

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 1 |
| Death sequences | 0 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 1 | 1 | 8 | 1 | 0 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:3208 — passive_run=310f mid_dist=0.0
- **passivity_window** r2.slp:4043 — passive_run=419f mid_dist=8.3
- **dropped_punish** r2.slp:6722 — opening@6722 start_pct=60.6 window=120f
- **neutral_loss** r2.slp:314 — hit@404 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r2.slp:1935 — hit@2025 opener=p2_action44 dist_90f_prior=7.5
- **neutral_loss** r2.slp:2847 — hit@2937 opener=p2_action44 dist_90f_prior=10.4
- **neutral_loss** r2.slp:3394 — hit@3484 opener=p2_action44 dist_90f_prior=0.1
- **neutral_loss** r2.slp:3666 — hit@3756 opener=p2_action44 dist_90f_prior=11.9
- **neutral_loss** r2.slp:6308 — hit@6398 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r2.slp:6735 — hit@6825 opener=p2_action57 dist_90f_prior=24.7

Appended 10 new gap(s) to `scenarios/gaps.json` (455 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, passivity.
