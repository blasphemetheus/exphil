# Coach Report — 20260829_231246

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.47 |
| Approaches (total) | 5 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 1.47 | 5 | 2 | 11 | 3 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:353 — passive_run=317f mid_dist=6.6
- **dropped_punish** r1.slp:2004 — opening@2004 start_pct=168.0 window=120f
- **dropped_punish** r1.slp:2870 — opening@2870 start_pct=13.0 window=120f
- **dropped_punish** r1.slp:5677 — opening@5677 start_pct=162.9 window=120f
- **neutral_loss** r1.slp:555 — hit@645 opener=p2_action60 dist_90f_prior=27.0
- **neutral_loss** r1.slp:1000 — hit@1090 opener=p2_action44 dist_90f_prior=22.9
- **neutral_loss** r1.slp:1666 — hit@1756 opener=p2_action44 dist_90f_prior=47.9
- **neutral_loss** r1.slp:2462 — hit@2552 opener=p2_action60 dist_90f_prior=6.7
- **neutral_loss** r1.slp:3677 — hit@3767 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r1.slp:4545 — hit@4635 opener=p2_action44 dist_90f_prior=39.5

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
