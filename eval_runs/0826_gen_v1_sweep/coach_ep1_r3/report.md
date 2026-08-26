# Coach Report — 20260826_123016

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 7 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 4 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 7 | 3 | 9 | 4 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:577 — passive_run=385f mid_dist=7.2
- **dropped_punish** r3.slp:1906 — opening@1906 start_pct=107.7 window=120f
- **dropped_punish** r3.slp:5135 — opening@5135 start_pct=181.6 window=120f
- **dropped_punish** r3.slp:5445 — opening@5445 start_pct=191.0 window=120f
- **dropped_punish** r3.slp:6017 — opening@6017 start_pct=206.0 window=120f
- **neutral_loss** r3.slp:430 — hit@520 opener=p2_action57 dist_90f_prior=55.7
- **neutral_loss** r3.slp:855 — hit@945 opener=p2_action44 dist_90f_prior=24.9
- **neutral_loss** r3.slp:1560 — hit@1650 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r3.slp:2646 — hit@2736 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r3.slp:3001 — hit@3091 opener=p2_action44 dist_90f_prior=6.6

Appended 10 new gap(s) to `scenarios/gaps.json` (256 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
