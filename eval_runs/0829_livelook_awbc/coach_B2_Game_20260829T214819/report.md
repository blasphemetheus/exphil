# Coach Report — 20260829_224705

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.08 |
| Approaches (total) | 9 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 2 |
| Dropped punishes | 2 |
| Death sequences | 4 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T214819.slp | 1.08 | 9 | 2 | 2 | 2 | 4 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T214819.slp:768 — passive_run=394f mid_dist=14.7
- **passivity_window** Game_20260829T214819.slp:2468 — passive_run=389f mid_dist=1.2
- **dropped_punish** Game_20260829T214819.slp:910 — opening@910 start_pct=29.4 window=120f
- **dropped_punish** Game_20260829T214819.slp:3211 — opening@3211 start_pct=55.6 window=120f
- **neutral_loss** Game_20260829T214819.slp:1347 — hit@1437 opener=p2_action63 dist_90f_prior=38.3
- **neutral_loss** Game_20260829T214819.slp:2121 — hit@2211 opener=p2_action68 dist_90f_prior=43.2
- **death_sequence** Game_20260829T214819.slp:1195 — death@1196 elapsed=1f opener=p2_action354
- **death_sequence** Game_20260829T214819.slp:1946 — death@1947 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260829T214819.slp:3078 — death@3079 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260829T214819.slp:3338 — death@3339 elapsed=1f opener=p2_action27

Appended 10 new gap(s) to `scenarios/gaps.json` (1753 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
