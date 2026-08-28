# Coach Report — 20260828_162805

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 4 |
| Death sequences | 0 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 0.49 | 3 | 0 | 11 | 4 | 0 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r5.slp:4916 — passive_run=460f mid_dist=8.3
- **passivity_window** r5.slp:5382 — passive_run=536f mid_dist=0.0
- **passivity_window** r5.slp:6013 — passive_run=584f mid_dist=8.3
- **dropped_punish** r5.slp:619 — opening@619 start_pct=21.9 window=120f
- **dropped_punish** r5.slp:1416 — opening@1416 start_pct=60.0 window=120f
- **dropped_punish** r5.slp:3327 — opening@3327 start_pct=154.0 window=120f
- **dropped_punish** r5.slp:4437 — opening@4437 start_pct=216.2 window=120f
- **neutral_loss** r5.slp:1775 — hit@1865 opener=p2_action63 dist_90f_prior=10.2
- **neutral_loss** r5.slp:2478 — hit@2568 opener=p2_action44 dist_90f_prior=32.1
- **neutral_loss** r5.slp:2883 — hit@2973 opener=p2_action44 dist_90f_prior=16.9

Appended 10 new gap(s) to `scenarios/gaps.json` (921 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, passivity.
