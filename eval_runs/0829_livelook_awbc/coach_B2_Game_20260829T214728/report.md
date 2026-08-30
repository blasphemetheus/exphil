# Coach Report — 20260829_224704

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 6 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 1 |
| Dropped punishes | 2 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T214728.slp | 0.0 | 6 | 2 | 1 | 2 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T214728.slp:1409 — opening@1409 start_pct=44.2 window=120f
- **dropped_punish** Game_20260829T214728.slp:2508 — opening@2508 start_pct=71.8 window=120f
- **neutral_loss** Game_20260829T214728.slp:436 — hit@526 opener=p2_action68 dist_90f_prior=28.3
- **death_sequence** Game_20260829T214728.slp:1220 — death@1221 elapsed=1f opener=p2_action26
- **death_sequence** Game_20260829T214728.slp:1653 — death@1654 elapsed=1f opener=p2_action20
- **death_sequence** Game_20260829T214728.slp:2668 — death@2669 elapsed=1f opener=p2_action12

Appended 6 new gap(s) to `scenarios/gaps.json` (1743 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
