# Coach Report — 20260829_150156

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 4 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r7.slp | 0.0 | 2 | 0 | 8 | 4 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r7.slp:3344 — passive_run=315f mid_dist=6.9
- **dropped_punish** r7.slp:2866 — opening@2866 start_pct=126.9 window=120f
- **dropped_punish** r7.slp:5688 — opening@5688 start_pct=35.8 window=120f
- **dropped_punish** r7.slp:6165 — opening@6165 start_pct=44.0 window=120f
- **dropped_punish** r7.slp:7277 — opening@7277 start_pct=118.1 window=120f
- **neutral_loss** r7.slp:883 — hit@973 opener=p2_action44 dist_90f_prior=55.5
- **neutral_loss** r7.slp:1664 — hit@1754 opener=p2_action56 dist_90f_prior=6.4
- **neutral_loss** r7.slp:1991 — hit@2081 opener=p2_action44 dist_90f_prior=5.4
- **neutral_loss** r7.slp:3133 — hit@3223 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r7.slp:3462 — hit@3552 opener=p2_action44 dist_90f_prior=6.9

Appended 10 new gap(s) to `scenarios/gaps.json` (1569 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
