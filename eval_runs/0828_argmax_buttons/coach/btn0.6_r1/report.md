# Coach Report — 20260828_162815

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 5 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 6 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.98 | 5 | 3 | 12 | 6 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:1298 — passive_run=378f mid_dist=8.5
- **dropped_punish** r1.slp:2975 — opening@2975 start_pct=147.0 window=120f
- **dropped_punish** r1.slp:3923 — opening@3923 start_pct=0.0 window=120f
- **dropped_punish** r1.slp:4281 — opening@4281 start_pct=17.6 window=120f
- **dropped_punish** r1.slp:5413 — opening@5413 start_pct=77.5 window=120f
- **dropped_punish** r1.slp:5938 — opening@5938 start_pct=92.6 window=120f
- **dropped_punish** r1.slp:6573 — opening@6573 start_pct=97.6 window=120f
- **neutral_loss** r1.slp:366 — hit@456 opener=p2_action44 dist_90f_prior=32.0
- **neutral_loss** r1.slp:785 — hit@875 opener=p2_action44 dist_90f_prior=7.4
- **neutral_loss** r1.slp:1274 — hit@1364 opener=p2_action53 dist_90f_prior=53.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1001 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
