# Coach Report — 20260828_171133

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.42 |
| Approaches (total) | 11 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 12 |
| Dropped punishes | 3 |
| Death sequences | 4 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T170020.slp | 0.42 | 11 | 2 | 12 | 3 | 4 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T170020.slp:4220 — passive_run=410f mid_dist=25.9
- **passivity_window** Game_20260828T170020.slp:5357 — passive_run=337f mid_dist=24.5
- **dropped_punish** Game_20260828T170020.slp:1642 — opening@1642 start_pct=30.1 window=120f
- **dropped_punish** Game_20260828T170020.slp:3898 — opening@3898 start_pct=100.8 window=120f
- **dropped_punish** Game_20260828T170020.slp:7207 — opening@7207 start_pct=139.4 window=120f
- **neutral_loss** Game_20260828T170020.slp:1318 — hit@1408 opener=p2_action53 dist_90f_prior=11.3
- **neutral_loss** Game_20260828T170020.slp:1662 — hit@1752 opener=p2_action57 dist_90f_prior=7.5
- **neutral_loss** Game_20260828T170020.slp:1920 — hit@2010 opener=p2_action53 dist_90f_prior=44.9
- **neutral_loss** Game_20260828T170020.slp:2913 — hit@3003 opener=p2_action69 dist_90f_prior=62.7
- **neutral_loss** Game_20260828T170020.slp:3484 — hit@3574 opener=p2_action20 dist_90f_prior=61.6

Appended 10 new gap(s) to `scenarios/gaps.json` (1131 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
