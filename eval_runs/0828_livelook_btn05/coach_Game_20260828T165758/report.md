# Coach Report — 20260828_171132

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 6 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 3 |
| Death sequences | 3 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T165758.slp | 0.0 | 6 | 2 | 12 | 3 | 3 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T165758.slp:1831 — passive_run=306f mid_dist=24.1
- **passivity_window** Game_20260828T165758.slp:4976 — passive_run=351f mid_dist=17.1
- **passivity_window** Game_20260828T165758.slp:5682 — passive_run=556f mid_dist=26.3
- **passivity_window** Game_20260828T165758.slp:7482 — passive_run=338f mid_dist=9.5
- **dropped_punish** Game_20260828T165758.slp:552 — opening@552 start_pct=43.8 window=120f
- **dropped_punish** Game_20260828T165758.slp:3280 — opening@3280 start_pct=100.7 window=120f
- **dropped_punish** Game_20260828T165758.slp:6234 — opening@6234 start_pct=156.8 window=120f
- **neutral_loss** Game_20260828T165758.slp:590 — hit@680 opener=p2_action66 dist_90f_prior=46.6
- **neutral_loss** Game_20260828T165758.slp:1106 — hit@1196 opener=p2_action66 dist_90f_prior=91.3
- **neutral_loss** Game_20260828T165758.slp:1625 — hit@1715 opener=p2_action66 dist_90f_prior=1.9

Appended 10 new gap(s) to `scenarios/gaps.json` (1121 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
