# Coach Report — 20260829_231239

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 4 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 4 |
| Death sequences | 3 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r6.slp | 0.0 | 4 | 0 | 7 | 4 | 3 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r6.slp:1669 — passive_run=409f mid_dist=13.3
- **passivity_window** r6.slp:2775 — passive_run=495f mid_dist=8.3
- **passivity_window** r6.slp:6847 — passive_run=335f mid_dist=3.5
- **dropped_punish** r6.slp:761 — opening@761 start_pct=77.0 window=120f
- **dropped_punish** r6.slp:1205 — opening@1205 start_pct=84.0 window=120f
- **dropped_punish** r6.slp:3248 — opening@3248 start_pct=190.0 window=120f
- **dropped_punish** r6.slp:3582 — opening@3582 start_pct=208.0 window=120f
- **neutral_loss** r6.slp:511 — hit@601 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r6.slp:1352 — hit@1442 opener=p2_action44 dist_90f_prior=15.4
- **neutral_loss** r6.slp:1950 — hit@2040 opener=p2_action48 dist_90f_prior=13.8

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
