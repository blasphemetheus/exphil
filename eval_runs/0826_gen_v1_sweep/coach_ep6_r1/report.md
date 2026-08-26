# Coach Report — 20260826_131028

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 2 | 0 | 10 | 3 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:3886 — passive_run=351f mid_dist=8.3
- **dropped_punish** r1.slp:2230 — opening@2230 start_pct=64.0 window=120f
- **dropped_punish** r1.slp:6243 — opening@6243 start_pct=36.8 window=120f
- **dropped_punish** r1.slp:6969 — opening@6969 start_pct=89.6 window=120f
- **neutral_loss** r1.slp:451 — hit@541 opener=p2_action44 dist_90f_prior=7.2
- **neutral_loss** r1.slp:1296 — hit@1386 opener=p2_action44 dist_90f_prior=9.7
- **neutral_loss** r1.slp:1638 — hit@1728 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r1.slp:2326 — hit@2416 opener=p2_action44 dist_90f_prior=8.0
- **neutral_loss** r1.slp:2957 — hit@3047 opener=p2_action60 dist_90f_prior=27.1
- **neutral_loss** r1.slp:3594 — hit@3684 opener=p2_action48 dist_90f_prior=25.4

Appended 10 new gap(s) to `scenarios/gaps.json` (385 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
