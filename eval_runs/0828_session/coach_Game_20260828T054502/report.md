# Coach Report — 20260828_171158

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 3 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 1 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T054502.slp | 0.0 | 3 | 1 | 6 | 1 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T054502.slp:834 — opening@834 start_pct=20.7 window=120f
- **neutral_loss** Game_20260828T054502.slp:359 — hit@449 opener=p2_action215 dist_90f_prior=55.3
- **neutral_loss** Game_20260828T054502.slp:1161 — hit@1251 opener=p2_action56 dist_90f_prior=40.9
- **neutral_loss** Game_20260828T054502.slp:1428 — hit@1518 opener=p2_action63 dist_90f_prior=9.3
- **neutral_loss** Game_20260828T054502.slp:1922 — hit@2012 opener=p2_action213 dist_90f_prior=70.1
- **neutral_loss** Game_20260828T054502.slp:2416 — hit@2506 opener=p2_action352 dist_90f_prior=6.8
- **neutral_loss** Game_20260828T054502.slp:2802 — hit@2892 opener=p2_action44 dist_90f_prior=36.5
- **death_sequence** Game_20260828T054502.slp:645 — death@646 elapsed=1f opener=p2_action25
- **death_sequence** Game_20260828T054502.slp:1517 — death@1625 elapsed=108f opener=p2_action63
- **death_sequence** Game_20260828T054502.slp:3391 — death@3392 elapsed=1f opener=p2_action252

Appended 10 new gap(s) to `scenarios/gaps.json` (1229 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
