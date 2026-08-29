# Coach Report — 20260829_120252

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 7 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 4 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r7.slp | 0.98 | 7 | 1 | 4 | 5 | 2 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r7.slp:3887 — passive_run=343f mid_dist=8.3
- **passivity_window** r7.slp:5578 — passive_run=433f mid_dist=8.3
- **passivity_window** r7.slp:6143 — passive_run=423f mid_dist=20.8
- **dropped_punish** r7.slp:819 — opening@819 start_pct=38.8 window=120f
- **dropped_punish** r7.slp:1149 — opening@1149 start_pct=48.8 window=120f
- **dropped_punish** r7.slp:2489 — opening@2489 start_pct=102.0 window=120f
- **dropped_punish** r7.slp:2853 — opening@2853 start_pct=106.2 window=120f
- **dropped_punish** r7.slp:3179 — opening@3179 start_pct=108.2 window=120f
- **neutral_loss** r7.slp:373 — hit@463 opener=p2_action56 dist_90f_prior=2.6
- **neutral_loss** r7.slp:1510 — hit@1600 opener=p2_action57 dist_90f_prior=84.7

Appended 10 new gap(s) to `scenarios/gaps.json` (1409 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
