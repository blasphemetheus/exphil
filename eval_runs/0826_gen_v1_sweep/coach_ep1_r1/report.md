# Coach Report — 20260826_123014

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 3 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.98 | 5 | 1 | 10 | 3 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:670 — passive_run=314f mid_dist=7.1
- **passivity_window** r1.slp:2537 — passive_run=311f mid_dist=7.0
- **dropped_punish** r1.slp:467 — opening@467 start_pct=43.8 window=120f
- **dropped_punish** r1.slp:4624 — opening@4624 start_pct=172.8 window=120f
- **dropped_punish** r1.slp:6674 — opening@6674 start_pct=255.8 window=120f
- **neutral_loss** r1.slp:861 — hit@951 opener=p2_action60 dist_90f_prior=0.7
- **neutral_loss** r1.slp:1250 — hit@1340 opener=p2_action44 dist_90f_prior=40.8
- **neutral_loss** r1.slp:1502 — hit@1592 opener=p2_action44 dist_90f_prior=6.4
- **neutral_loss** r1.slp:2734 — hit@2824 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r1.slp:3974 — hit@4064 opener=p2_action60 dist_90f_prior=28.8

Appended 10 new gap(s) to `scenarios/gaps.json` (236 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
