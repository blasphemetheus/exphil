# Coach Report — 20260829_224651

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 2 |
| Dropped punishes | 1 |
| Death sequences | 3 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T220134.slp | 0.0 | 4 | 2 | 2 | 1 | 3 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T220134.slp:1480 — passive_run=305f mid_dist=22.5
- **dropped_punish** Game_20260829T220134.slp:366 — opening@366 start_pct=9.5 window=120f
- **neutral_loss** Game_20260829T220134.slp:1390 — hit@1480 opener=p2_action68 dist_90f_prior=32.2
- **neutral_loss** Game_20260829T220134.slp:2083 — hit@2173 opener=p2_action360 dist_90f_prior=19.8
- **death_sequence** Game_20260829T220134.slp:1113 — death@1114 elapsed=1f opener=p2_action25
- **death_sequence** Game_20260829T220134.slp:1725 — death@1726 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260829T220134.slp:2240 — death@2241 elapsed=1f opener=p2_action20

Appended 7 new gap(s) to `scenarios/gaps.json` (1674 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
