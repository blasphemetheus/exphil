# Coach Report — 20260829_120240

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 4 |
| Death sequences | 2 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.49 | 3 | 0 | 10 | 4 | 2 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:1513 — passive_run=306f mid_dist=8.3
- **passivity_window** r1.slp:3970 — passive_run=340f mid_dist=8.3
- **passivity_window** r1.slp:5137 — passive_run=423f mid_dist=8.3
- **dropped_punish** r1.slp:2034 — opening@2034 start_pct=88.8 window=120f
- **dropped_punish** r1.slp:2313 — opening@2313 start_pct=93.4 window=120f
- **dropped_punish** r1.slp:3753 — opening@3753 start_pct=140.6 window=120f
- **dropped_punish** r1.slp:4930 — opening@4930 start_pct=179.8 window=120f
- **neutral_loss** r1.slp:380 — hit@470 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r1.slp:1191 — hit@1281 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r1.slp:1797 — hit@1887 opener=p2_action63 dist_90f_prior=2.0

Appended 10 new gap(s) to `scenarios/gaps.json` (1269 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
