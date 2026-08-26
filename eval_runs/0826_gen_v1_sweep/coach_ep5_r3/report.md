# Coach Report — 20260826_130212

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 5 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 5 | 3 | 10 | 3 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:2264 — passive_run=556f mid_dist=24.5
- **passivity_window** r3.slp:6977 — passive_run=378f mid_dist=8.3
- **dropped_punish** r3.slp:3747 — opening@3747 start_pct=12.0 window=120f
- **dropped_punish** r3.slp:4375 — opening@4375 start_pct=58.1 window=120f
- **dropped_punish** r3.slp:6177 — opening@6177 start_pct=118.2 window=120f
- **neutral_loss** r3.slp:478 — hit@568 opener=p2_action60 dist_90f_prior=9.9
- **neutral_loss** r3.slp:843 — hit@933 opener=p2_action60 dist_90f_prior=54.8
- **neutral_loss** r3.slp:1546 — hit@1636 opener=p2_action57 dist_90f_prior=26.6
- **neutral_loss** r3.slp:2160 — hit@2250 opener=p2_action44 dist_90f_prior=0.4
- **neutral_loss** r3.slp:2665 — hit@2755 opener=p2_action44 dist_90f_prior=7.2

Appended 10 new gap(s) to `scenarios/gaps.json` (375 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
