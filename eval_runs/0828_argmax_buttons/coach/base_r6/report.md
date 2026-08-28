# Coach Report — 20260828_162806

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 3 |
| Death sequences | 3 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r6.slp | 0.49 | 3 | 0 | 9 | 3 | 3 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r6.slp:3767 — passive_run=372f mid_dist=8.3
- **passivity_window** r6.slp:4433 — passive_run=392f mid_dist=8.3
- **passivity_window** r6.slp:4832 — passive_run=477f mid_dist=8.3
- **passivity_window** r6.slp:5315 — passive_run=515f mid_dist=8.3
- **dropped_punish** r6.slp:716 — opening@716 start_pct=38.7 window=120f
- **dropped_punish** r6.slp:1651 — opening@1651 start_pct=56.7 window=120f
- **dropped_punish** r6.slp:3040 — opening@3040 start_pct=117.8 window=120f
- **neutral_loss** r6.slp:462 — hit@552 opener=p2_action44 dist_90f_prior=5.2
- **neutral_loss** r6.slp:812 — hit@902 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r6.slp:1290 — hit@1380 opener=p2_action48 dist_90f_prior=0.5

Appended 10 new gap(s) to `scenarios/gaps.json` (931 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
