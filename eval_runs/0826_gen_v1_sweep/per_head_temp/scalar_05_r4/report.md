# Coach Report — 20260827_180451

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.98 | 4 | 0 | 9 | 3 | 2 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:1655 — passive_run=318f mid_dist=8.3
- **passivity_window** r4.slp:2850 — passive_run=360f mid_dist=8.3
- **passivity_window** r4.slp:3468 — passive_run=357f mid_dist=7.1
- **passivity_window** r4.slp:6246 — passive_run=310f mid_dist=14.8
- **dropped_punish** r4.slp:2156 — opening@2156 start_pct=145.7 window=120f
- **dropped_punish** r4.slp:4158 — opening@4158 start_pct=208.0 window=120f
- **dropped_punish** r4.slp:6980 — opening@6980 start_pct=100.2 window=120f
- **neutral_loss** r4.slp:1296 — hit@1386 opener=p2_action48 dist_90f_prior=20.5
- **neutral_loss** r4.slp:1862 — hit@1952 opener=p2_action60 dist_90f_prior=19.7
- **neutral_loss** r4.slp:2714 — hit@2804 opener=p2_action44 dist_90f_prior=7.0

Appended 10 new gap(s) to `scenarios/gaps.json` (564 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
