# Coach Report — 20260828_162816

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 5 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.0 | 2 | 0 | 10 | 5 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:3762 — passive_run=302f mid_dist=8.3
- **passivity_window** r2.slp:4780 — passive_run=628f mid_dist=8.3
- **dropped_punish** r2.slp:361 — opening@361 start_pct=13.2 window=120f
- **dropped_punish** r2.slp:2837 — opening@2837 start_pct=133.8 window=120f
- **dropped_punish** r2.slp:5415 — opening@5415 start_pct=273.5 window=120f
- **dropped_punish** r2.slp:6754 — opening@6754 start_pct=75.9 window=120f
- **dropped_punish** r2.slp:7240 — opening@7240 start_pct=118.1 window=120f
- **neutral_loss** r2.slp:499 — hit@589 opener=p2_action57 dist_90f_prior=6.9
- **neutral_loss** r2.slp:842 — hit@932 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r2.slp:1711 — hit@1801 opener=p2_action60 dist_90f_prior=38.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1011 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
