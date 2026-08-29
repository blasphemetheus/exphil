# Coach Report — 20260829_120244

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 6 |
| Death sequences | 3 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r6.slp | 0.49 | 3 | 1 | 7 | 6 | 3 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r6.slp:6349 — passive_run=345f mid_dist=7.0
- **dropped_punish** r6.slp:944 — opening@944 start_pct=52.4 window=120f
- **dropped_punish** r6.slp:1575 — opening@1575 start_pct=100.2 window=120f
- **dropped_punish** r6.slp:2900 — opening@2900 start_pct=168.6 window=120f
- **dropped_punish** r6.slp:4240 — opening@4240 start_pct=195.2 window=120f
- **dropped_punish** r6.slp:4792 — opening@4792 start_pct=199.2 window=120f
- **dropped_punish** r6.slp:5369 — opening@5369 start_pct=223.6 window=120f
- **neutral_loss** r6.slp:2128 — hit@2218 opener=p2_action57 dist_90f_prior=22.2
- **neutral_loss** r6.slp:2515 — hit@2605 opener=p2_action60 dist_90f_prior=38.7
- **neutral_loss** r6.slp:3185 — hit@3275 opener=p2_action44 dist_90f_prior=1.2

Appended 10 new gap(s) to `scenarios/gaps.json` (1319 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
