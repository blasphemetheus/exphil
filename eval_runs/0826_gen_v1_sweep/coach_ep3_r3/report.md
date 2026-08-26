# Coach Report — 20260826_124538

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 1 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 2 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 1 | 0 | 8 | 2 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:5268 — passive_run=343f mid_dist=8.3
- **dropped_punish** r3.slp:1932 — opening@1932 start_pct=121.9 window=120f
- **dropped_punish** r3.slp:4802 — opening@4802 start_pct=37.6 window=120f
- **neutral_loss** r3.slp:807 — hit@897 opener=p2_action44 dist_90f_prior=53.3
- **neutral_loss** r3.slp:1682 — hit@1772 opener=p2_action60 dist_90f_prior=57.1
- **neutral_loss** r3.slp:3065 — hit@3155 opener=p2_action44 dist_90f_prior=0.0
- **neutral_loss** r3.slp:3551 — hit@3641 opener=p2_action44 dist_90f_prior=48.7
- **neutral_loss** r3.slp:4297 — hit@4387 opener=p2_action44 dist_90f_prior=63.3
- **neutral_loss** r3.slp:5009 — hit@5099 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r3.slp:5809 — hit@5899 opener=p2_action44 dist_90f_prior=6.7

Appended 10 new gap(s) to `scenarios/gaps.json` (315 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
