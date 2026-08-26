# Coach Report — 20260826_131028

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 2 |
| Death sequences | 3 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.98 | 4 | 2 | 9 | 2 | 3 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:4613 — passive_run=530f mid_dist=8.3
- **passivity_window** r2.slp:5207 — passive_run=602f mid_dist=8.3
- **dropped_punish** r2.slp:3029 — opening@3029 start_pct=151.7 window=120f
- **dropped_punish** r2.slp:6420 — opening@6420 start_pct=14.4 window=120f
- **neutral_loss** r2.slp:421 — hit@511 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r2.slp:832 — hit@922 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r2.slp:1845 — hit@1935 opener=p2_action57 dist_90f_prior=66.1
- **neutral_loss** r2.slp:2227 — hit@2317 opener=p2_action44 dist_90f_prior=39.9
- **neutral_loss** r2.slp:3795 — hit@3885 opener=p2_action60 dist_90f_prior=56.9
- **neutral_loss** r2.slp:4082 — hit@4172 opener=p2_action44 dist_90f_prior=7.0

Appended 10 new gap(s) to `scenarios/gaps.json` (395 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
