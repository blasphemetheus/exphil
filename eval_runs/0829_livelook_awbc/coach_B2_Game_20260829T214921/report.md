# Coach Report — 20260829_224707

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.14 |
| Approaches (total) | 7 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 3 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T214921.slp | 1.14 | 7 | 3 | 5 | 3 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260829T214921.slp:367 — opening@367 start_pct=8.0 window=120f
- **dropped_punish** Game_20260829T214921.slp:4540 — opening@4540 start_pct=28.0 window=120f
- **dropped_punish** Game_20260829T214921.slp:6180 — opening@6180 start_pct=85.4 window=120f
- **neutral_loss** Game_20260829T214921.slp:2263 — hit@2353 opener=p2_action215 dist_90f_prior=57.4
- **neutral_loss** Game_20260829T214921.slp:2875 — hit@2965 opener=p2_action215 dist_90f_prior=52.3
- **neutral_loss** Game_20260829T214921.slp:3558 — hit@3648 opener=p2_action63 dist_90f_prior=12.5
- **neutral_loss** Game_20260829T214921.slp:4542 — hit@4632 opener=p2_action69 dist_90f_prior=2.5
- **neutral_loss** Game_20260829T214921.slp:5277 — hit@5367 opener=p2_action68 dist_90f_prior=24.9
- **death_sequence** Game_20260829T214921.slp:2068 — death@2069 elapsed=1f opener=p2_action25
- **death_sequence** Game_20260829T214921.slp:3258 — death@3259 elapsed=1f opener=p2_action369

Appended 10 new gap(s) to `scenarios/gaps.json` (1763 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
