# Coach Report — 20260829_150153

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 7 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.49 | 4 | 1 | 6 | 7 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:2067 — passive_run=432f mid_dist=29.0
- **passivity_window** r4.slp:3891 — passive_run=404f mid_dist=8.3
- **dropped_punish** r4.slp:1153 — opening@1153 start_pct=54.0 window=120f
- **dropped_punish** r4.slp:1507 — opening@1507 start_pct=62.0 window=120f
- **dropped_punish** r4.slp:2465 — opening@2465 start_pct=92.3 window=120f
- **dropped_punish** r4.slp:3635 — opening@3635 start_pct=155.3 window=120f
- **dropped_punish** r4.slp:5006 — opening@5006 start_pct=5.0 window=120f
- **dropped_punish** r4.slp:5677 — opening@5677 start_pct=48.4 window=120f
- **dropped_punish** r4.slp:7037 — opening@7037 start_pct=84.1 window=120f
- **neutral_loss** r4.slp:747 — hit@837 opener=p2_action44 dist_90f_prior=6.8

Appended 10 new gap(s) to `scenarios/gaps.json` (1539 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
