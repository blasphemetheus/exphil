# Coach Report — 20260829_231225

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.49 | 2 | 1 | 12 | 2 | 1 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:1276 — passive_run=336f mid_dist=8.3
- **passivity_window** r1.slp:1618 — passive_run=375f mid_dist=8.3
- **passivity_window** r1.slp:5762 — passive_run=336f mid_dist=7.1
- **passivity_window** r1.slp:6484 — passive_run=341f mid_dist=6.8
- **dropped_punish** r1.slp:4277 — opening@4277 start_pct=9.5 window=120f
- **dropped_punish** r1.slp:4667 — opening@4667 start_pct=13.7 window=120f
- **neutral_loss** r1.slp:374 — hit@464 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r1.slp:1167 — hit@1257 opener=p2_action44 dist_90f_prior=56.8
- **neutral_loss** r1.slp:2305 — hit@2395 opener=p2_action60 dist_90f_prior=27.8
- **neutral_loss** r1.slp:2946 — hit@3036 opener=p2_action44 dist_90f_prior=7.1

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
