# Coach Report — 20260826_122127

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.0 |
| Approaches (total) | 4 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 3 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 1.0 | 4 | 3 | 7 | 3 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:4890 — passive_run=459f mid_dist=8.3
- **dropped_punish** r1.slp:2345 — opening@2345 start_pct=88.9 window=120f
- **dropped_punish** r1.slp:2601 — opening@2601 start_pct=93.9 window=120f
- **dropped_punish** r1.slp:3187 — opening@3187 start_pct=127.7 window=120f
- **neutral_loss** r1.slp:1345 — hit@1435 opener=p2_action60 dist_90f_prior=6.7
- **neutral_loss** r1.slp:1711 — hit@1801 opener=p2_action63 dist_90f_prior=32.7
- **neutral_loss** r1.slp:2662 — hit@2752 opener=p2_action44 dist_90f_prior=36.8
- **neutral_loss** r1.slp:3232 — hit@3322 opener=p2_action63 dist_90f_prior=57.5
- **neutral_loss** r1.slp:3542 — hit@3632 opener=p2_action57 dist_90f_prior=87.3
- **neutral_loss** r1.slp:4021 — hit@4111 opener=p2_action44 dist_90f_prior=20.4

Appended 10 new gap(s) to `scenarios/gaps.json` (226 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
