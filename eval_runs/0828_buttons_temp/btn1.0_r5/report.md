# Coach Report — 20260828_121907

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 6 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.49 | 3 | 1 | 8 | 3 | 2 | 6 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:1501 — passive_run=338f mid_dist=6.4
- **passivity_window** r5.slp:2970 — passive_run=310f mid_dist=8.3
- **passivity_window** r5.slp:3693 — passive_run=382f mid_dist=8.3
- **passivity_window** r5.slp:5185 — passive_run=454f mid_dist=8.3
- **passivity_window** r5.slp:5693 — passive_run=544f mid_dist=8.3
- **passivity_window** r5.slp:6244 — passive_run=777f mid_dist=8.3
- **dropped_punish** r5.slp:667 — opening@667 start_pct=62.4 window=120f
- **dropped_punish** r5.slp:2340 — opening@2340 start_pct=110.2 window=120f
- **dropped_punish** r5.slp:4634 — opening@4634 start_pct=217.3 window=120f
- **neutral_loss** r5.slp:394 — hit@484 opener=p2_action44 dist_90f_prior=6.9

Appended 10 new gap(s) to `scenarios/gaps.json` (721 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
