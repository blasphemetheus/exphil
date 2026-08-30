# Coach Report — 20260829_224647

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 9 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 1 |
| Death sequences | 3 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T215746.slp | 0.0 | 9 | 0 | 7 | 1 | 3 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T215746.slp:1286 — passive_run=543f mid_dist=23.4
- **passivity_window** Game_20260829T215746.slp:2141 — passive_run=494f mid_dist=16.6
- **passivity_window** Game_20260829T215746.slp:3298 — passive_run=608f mid_dist=6.8
- **dropped_punish** Game_20260829T215746.slp:396 — opening@396 start_pct=24.3 window=120f
- **neutral_loss** Game_20260829T215746.slp:472 — hit@562 opener=p2_action68 dist_90f_prior=33.7
- **neutral_loss** Game_20260829T215746.slp:1406 — hit@1496 opener=p2_action215 dist_90f_prior=16.9
- **neutral_loss** Game_20260829T215746.slp:2237 — hit@2327 opener=p2_action63 dist_90f_prior=8.0
- **neutral_loss** Game_20260829T215746.slp:2517 — hit@2607 opener=p2_action213 dist_90f_prior=21.4
- **neutral_loss** Game_20260829T215746.slp:3346 — hit@3436 opener=p2_action50 dist_90f_prior=11.4
- **neutral_loss** Game_20260829T215746.slp:4042 — hit@4132 opener=p2_action63 dist_90f_prior=47.0

Appended 10 new gap(s) to `scenarios/gaps.json` (1648 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
