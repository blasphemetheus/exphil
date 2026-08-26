# Coach Report — 20260826_125353

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 4 |
| Death sequences | 0 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 1 | 1 | 10 | 4 | 0 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:2522 — passive_run=372f mid_dist=20.5
- **passivity_window** r1.slp:6635 — passive_run=434f mid_dist=0.0
- **dropped_punish** r1.slp:772 — opening@772 start_pct=42.5 window=120f
- **dropped_punish** r1.slp:1807 — opening@1807 start_pct=76.2 window=120f
- **dropped_punish** r1.slp:2141 — opening@2141 start_pct=96.8 window=120f
- **dropped_punish** r1.slp:2383 — opening@2383 start_pct=101.8 window=120f
- **neutral_loss** r1.slp:1500 — hit@1590 opener=p2_action44 dist_90f_prior=29.5
- **neutral_loss** r1.slp:2490 — hit@2580 opener=p2_action60 dist_90f_prior=60.6
- **neutral_loss** r1.slp:2793 — hit@2883 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r1.slp:3142 — hit@3232 opener=p2_action44 dist_90f_prior=6.9

Appended 10 new gap(s) to `scenarios/gaps.json` (325 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, passivity.
