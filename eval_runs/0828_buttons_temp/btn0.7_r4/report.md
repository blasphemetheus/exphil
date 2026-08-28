# Coach Report — 20260828_123231

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 5 |
| Conversions (total) | 2 |
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
| r4.slp | 0.98 | 5 | 2 | 8 | 4 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:5636 — passive_run=305f mid_dist=13.6
- **dropped_punish** r4.slp:777 — opening@777 start_pct=41.8 window=120f
- **dropped_punish** r4.slp:2805 — opening@2805 start_pct=170.2 window=120f
- **dropped_punish** r4.slp:6050 — opening@6050 start_pct=64.2 window=120f
- **dropped_punish** r4.slp:6334 — opening@6334 start_pct=82.7 window=120f
- **neutral_loss** r4.slp:1463 — hit@1553 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r4.slp:2170 — hit@2260 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r4.slp:2511 — hit@2601 opener=p2_action44 dist_90f_prior=9.5
- **neutral_loss** r4.slp:3202 — hit@3292 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r4.slp:3597 — hit@3687 opener=p2_action44 dist_90f_prior=6.7

Appended 10 new gap(s) to `scenarios/gaps.json` (761 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
