# Coach Report — 20260828_123229

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.49 | 2 | 0 | 12 | 2 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:5703 — passive_run=651f mid_dist=8.3
- **dropped_punish** r2.slp:1948 — opening@1948 start_pct=106.1 window=120f
- **dropped_punish** r2.slp:3796 — opening@3796 start_pct=251.0 window=120f
- **neutral_loss** r2.slp:519 — hit@609 opener=p2_action44 dist_90f_prior=19.8
- **neutral_loss** r2.slp:897 — hit@987 opener=p2_action60 dist_90f_prior=33.6
- **neutral_loss** r2.slp:1189 — hit@1279 opener=p2_action44 dist_90f_prior=9.0
- **neutral_loss** r2.slp:2018 — hit@2108 opener=p2_action44 dist_90f_prior=34.6
- **neutral_loss** r2.slp:2355 — hit@2445 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r2.slp:3112 — hit@3202 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r2.slp:3466 — hit@3556 opener=p2_action44 dist_90f_prior=71.5

Appended 10 new gap(s) to `scenarios/gaps.json` (741 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
