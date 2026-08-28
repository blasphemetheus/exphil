# Coach Report — 20260828_162804

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 1 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.49 | 3 | 2 | 8 | 1 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:2446 — passive_run=405f mid_dist=8.3
- **dropped_punish** r4.slp:4499 — opening@4499 start_pct=16.6 window=120f
- **neutral_loss** r4.slp:508 — hit@598 opener=p2_action44 dist_90f_prior=0.3
- **neutral_loss** r4.slp:893 — hit@983 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r4.slp:1423 — hit@1513 opener=p2_action44 dist_90f_prior=13.6
- **neutral_loss** r4.slp:3342 — hit@3432 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r4.slp:5042 — hit@5132 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r4.slp:5821 — hit@5911 opener=p2_action53 dist_90f_prior=43.3
- **neutral_loss** r4.slp:6249 — hit@6339 opener=p2_action44 dist_90f_prior=56.0
- **neutral_loss** r4.slp:6797 — hit@6887 opener=p2_action44 dist_90f_prior=0.0

Appended 10 new gap(s) to `scenarios/gaps.json` (911 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
