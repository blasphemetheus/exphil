# Coach Report — 20260829_224701

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 7 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T214420.slp | 0.0 | 7 | 0 | 5 | 1 | 4 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T214420.slp:6497 — passive_run=369f mid_dist=8.3
- **dropped_punish** Game_20260829T214420.slp:1702 — opening@1702 start_pct=72.3 window=120f
- **neutral_loss** Game_20260829T214420.slp:454 — hit@544 opener=p2_action57 dist_90f_prior=2.2
- **neutral_loss** Game_20260829T214420.slp:2457 — hit@2547 opener=p2_action66 dist_90f_prior=55.4
- **neutral_loss** Game_20260829T214420.slp:2997 — hit@3087 opener=p2_action215 dist_90f_prior=49.8
- **neutral_loss** Game_20260829T214420.slp:3685 — hit@3775 opener=p2_action65 dist_90f_prior=86.4
- **neutral_loss** Game_20260829T214420.slp:5640 — hit@5730 opener=p2_action69 dist_90f_prior=6.9
- **death_sequence** Game_20260829T214420.slp:1020 — death@1021 elapsed=1f opener=p2_action28
- **death_sequence** Game_20260829T214420.slp:2787 — death@2788 elapsed=1f opener=p2_action252
- **death_sequence** Game_20260829T214420.slp:4493 — death@4494 elapsed=1f opener=p2_action253

Appended 10 new gap(s) to `scenarios/gaps.json` (1730 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
