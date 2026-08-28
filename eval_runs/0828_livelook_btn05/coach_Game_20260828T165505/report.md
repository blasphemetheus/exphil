# Coach Report — 20260828_171130

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.08 |
| Approaches (total) | 11 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 4 |
| Death sequences | 4 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T165505.slp | 1.08 | 11 | 1 | 12 | 4 | 4 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T165505.slp:364 — passive_run=378f mid_dist=6.0
- **passivity_window** Game_20260828T165505.slp:6314 — passive_run=397f mid_dist=10.4
- **dropped_punish** Game_20260828T165505.slp:3827 — opening@3827 start_pct=80.2 window=120f
- **dropped_punish** Game_20260828T165505.slp:4444 — opening@4444 start_pct=85.0 window=120f
- **dropped_punish** Game_20260828T165505.slp:6102 — opening@6102 start_pct=95.0 window=120f
- **dropped_punish** Game_20260828T165505.slp:6699 — opening@6699 start_pct=109.0 window=120f
- **neutral_loss** Game_20260828T165505.slp:316 — hit@406 opener=p2_action66 dist_90f_prior=59.6
- **neutral_loss** Game_20260828T165505.slp:1709 — hit@1799 opener=p2_action57 dist_90f_prior=1.6
- **neutral_loss** Game_20260828T165505.slp:2534 — hit@2624 opener=p2_action66 dist_90f_prior=51.7
- **neutral_loss** Game_20260828T165505.slp:3419 — hit@3509 opener=p2_action66 dist_90f_prior=98.0

Appended 10 new gap(s) to `scenarios/gaps.json` (1111 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
