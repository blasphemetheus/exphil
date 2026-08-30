# Coach Report — 20260829_231256

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 0 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.0 | 0 | 0 | 11 | 5 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r8.slp:3598 — passive_run=407f mid_dist=7.1
- **dropped_punish** r8.slp:1476 — opening@1476 start_pct=98.1 window=120f
- **dropped_punish** r8.slp:2791 — opening@2791 start_pct=138.8 window=120f
- **dropped_punish** r8.slp:4588 — opening@4588 start_pct=27.9 window=120f
- **dropped_punish** r8.slp:4988 — opening@4988 start_pct=49.4 window=120f
- **dropped_punish** r8.slp:7095 — opening@7095 start_pct=168.9 window=120f
- **neutral_loss** r8.slp:526 — hit@616 opener=p2_action60 dist_90f_prior=37.4
- **neutral_loss** r8.slp:1162 — hit@1252 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r8.slp:1896 — hit@1986 opener=p2_action44 dist_90f_prior=28.8
- **neutral_loss** r8.slp:2469 — hit@2559 opener=p2_action57 dist_90f_prior=36.4

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
