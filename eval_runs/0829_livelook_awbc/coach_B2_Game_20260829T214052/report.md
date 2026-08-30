# Coach Report — 20260829_224658

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.45 |
| Approaches (total) | 11 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 3 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T214052.slp | 0.45 | 11 | 3 | 11 | 3 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T214052.slp:1191 — opening@1191 start_pct=13.0 window=120f
- **dropped_punish** Game_20260829T214052.slp:4968 — opening@4968 start_pct=75.7 window=120f
- **dropped_punish** Game_20260829T214052.slp:6444 — opening@6444 start_pct=119.0 window=120f
- **neutral_loss** Game_20260829T214052.slp:379 — hit@469 opener=p2_action215 dist_90f_prior=47.6
- **neutral_loss** Game_20260829T214052.slp:853 — hit@943 opener=p2_action56 dist_90f_prior=22.6
- **neutral_loss** Game_20260829T214052.slp:1252 — hit@1342 opener=p2_action69 dist_90f_prior=40.5
- **neutral_loss** Game_20260829T214052.slp:2574 — hit@2664 opener=p2_action67 dist_90f_prior=30.5
- **neutral_loss** Game_20260829T214052.slp:3998 — hit@4088 opener=p2_action215 dist_90f_prior=78.2
- **neutral_loss** Game_20260829T214052.slp:4342 — hit@4432 opener=p2_action56 dist_90f_prior=30.8
- **neutral_loss** Game_20260829T214052.slp:4593 — hit@4683 opener=p2_action56 dist_90f_prior=33.7

Appended 10 new gap(s) to `scenarios/gaps.json` (1710 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
