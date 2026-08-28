# Coach Report — 20260828_124555

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 5 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.98 | 5 | 2 | 7 | 5 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:2772 — passive_run=470f mid_dist=8.3
- **passivity_window** r4.slp:4947 — passive_run=325f mid_dist=19.1
- **dropped_punish** r4.slp:490 — opening@490 start_pct=67.3 window=120f
- **dropped_punish** r4.slp:1164 — opening@1164 start_pct=104.9 window=120f
- **dropped_punish** r4.slp:2297 — opening@2297 start_pct=162.8 window=120f
- **dropped_punish** r4.slp:5333 — opening@5333 start_pct=29.4 window=120f
- **dropped_punish** r4.slp:7212 — opening@7212 start_pct=106.6 window=120f
- **neutral_loss** r4.slp:1503 — hit@1593 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r4.slp:1891 — hit@1981 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r4.slp:3422 — hit@3512 opener=p2_action44 dist_90f_prior=7.1

Appended 10 new gap(s) to `scenarios/gaps.json` (811 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
